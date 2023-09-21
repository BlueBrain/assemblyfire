"""
Assembly detection related utility functions (mostly loading simulation related stuff)
author: András Ecker, last update: 06.2023
"""

import os
import json
import h5py
import warnings
from collections import namedtuple
import numpy as np
import pandas as pd
from libsonata import EdgeStorage
from bluepysnap import Circuit, Simulation

SpikeMatrixResult = namedtuple("SpikeMatrixResult", ["spike_matrix", "gids", "t_bins"])


def get_bluepy_circuit(circuitconfig_path):
    return Circuit(circuitconfig_path)


def get_bluepy_simulation(blueconfig_path):
    return Simulation(blueconfig_path)


def get_bglibpy_ssim(blueconfig_path):
    try:
        import bglibpy
    except ImportError as e:
        msg = ("Assemblyfire requirements are not installed.\n"
               "Please pip install bglibpy as follows:\n"
               " pip install -i https://bbpteam.epfl.ch/repository/devpi/bbprelman/dev/+simple/ bglibpy")
        raise ImportError(str(e) + "\n\n" + msg)
    return bglibpy.SSim(blueconfig_path, record_dt=0.1)


def ensure_dir(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


def get_sim_path(root_path):
    """Loads in simulation paths as pandas (MultiIndex) DataFrame generated by bbp-workflow"""
    pklf_name = os.path.join(root_path, "analyses", "simulations.pkl")
    sim_paths = pd.read_pickle(pklf_name)
    level_names = sim_paths.index.names
    assert len(level_names) == 1 and level_names[0] == "seed", "Only a campaign/DataFrame with single" \
           "`coord`/index level called `seed` is acceptable by assemblyfire"
    return sim_paths


def get_bluepy_circuit_from_root_path(root_path):
    """Return bluepy circuit from the first simulation in the project root"""
    return get_bluepy_simulation(get_sim_path(root_path).iloc[0]).circuit


def get_stimulus_stream(f_name, t_start=None, t_end=None):
    """Reads the series of presented patterns from .txt file"""
    stim_times, patterns = [], []
    with open(f_name, "r") as f:
        for line in f:
            tmp = line.strip().split()
            stim_times.append(float(tmp[0]))
            patterns.append(tmp[1])
    stim_times, patterns = np.asarray(stim_times), np.asarray(patterns)
    if t_start is None and t_end is None:  # TODO: handle them separately as well...
        return stim_times, patterns
    else:
        idx = np.where((t_start < stim_times) & (stim_times < t_end))[0]
        return stim_times[idx], patterns[idx]


def get_pattern_node_idx(jf_name):
    """Loads gids corresponding to patterns from JSON"""
    with open(jf_name, "r") as f:
        node_idx = json.load(f)
    return {pattern_name: np.array(tmp["node_id"]) for pattern_name, tmp in node_idx.items()}


def get_pattern_distance(locf_name, jf_name):
    """Gets Earth mover's distance of the input pattern fibers"""
    from scipy.spatial.distance import cdist
    from scipy.optimize import linear_sum_assignment
    tmp = np.loadtxt(locf_name)
    gids, pos = tmp[:, 0].astype(int), tmp[:, 1:]
    pattern_gids = get_pattern_node_idx(jf_name)
    pattern_pos = {pattern_name: pos[np.in1d(gids, gids_, assume_unique=True), :]
                   for pattern_name, gids_ in pattern_gids.items()}
    pattern_names = np.sort(list(pattern_pos.keys()))
    row_idx, col_idx = np.triu_indices(len(pattern_names), k=1)
    emd = np.zeros_like(row_idx, dtype=np.float32)
    for i, (row_id, col_id) in enumerate(zip(row_idx, col_idx)):
        dists = cdist(pattern_pos[pattern_names[row_id]], pattern_pos[pattern_names[col_id]])
        emd[i] = np.sum(dists[linear_sum_assignment(dists)]) / len(pattern_pos[pattern_names[row_id]])
    return pattern_names, emd


def get_node_idx(c, node_pop, target):
    return c.nodes[node_pop].ids(target)


def get_node_properties(c, node_pop, node_idx, properties):
    return c.nodes[node_pop].get(node_idx, properties)


def load_nrn_df(h5f_name, prefix="connectivity"):
    """Loads neuron locations in (supersampled) flat space
    (and a few extra stuff) from `conntility` object saved to HDF5"""
    df = pd.read_hdf(h5f_name, "%s/full_matrix/vertex_properties" % prefix)
    return df.drop(columns=["x", "y", "z"])


def _get_nrn_df(c, node_pop, target):
    """Gets neuron locations (and a few extra stuff) in (supersampled) flat space"""
    from conntility.circuit_models.neuron_groups import load_neurons
    df = load_neurons(c, ["layer", "x", "y", "z", "mtype", "ss_flat_x", "ss_flat_y", "depth"], target, node_pop)
    return df.set_index("node_ids").drop(columns=["x", "y", "z"])


def get_nrn_df(h5f_name, prefix, root_path, target, node_pop="S1nonbarrel_neurons"):
    """Tries to load neuron locations from saved connectivity matrix object, or calculates them if they aren't saved"""
    h5f = h5py.File(h5f_name, "r")
    if prefix in list(h5f.keys()):
        h5f.close()
        nrn_loc_df = load_nrn_df(h5f_name, prefix)
    else:
        nrn_loc_df = _get_nrn_df(get_bluepy_circuit_from_root_path(root_path), node_pop, target)
    return nrn_loc_df


def get_spikes(sim, node_pop, gids, t_start, t_end):
    """Extracts spikes (using bluepy)"""
    if gids is None:
        spikes = sim.spikes[node_pop].get(t_start=t_start, t_stop=t_end)
    else:
        spikes = sim.spikes[node_pop].get(gids, t_start=t_start, t_stop=t_end)
    return spikes.index.to_numpy(), spikes.to_numpy()


def get_proj_edge_pops(circuit_config, local_edge_pop):
    """Gets the name of projection (non-local edge)
    edge populations from `bluepysnap.Circuit.config` object"""
    return [list(tmp["populations"].keys())[0] for tmp in circuit_config["networks"]["edges"]
            if list(tmp["populations"].keys())[0] != local_edge_pop]


def _get_spikef_names(sim_config):
    """Gets the name of the SpikeFile from bluepy.Simulation.config object"""
    spikef_names = {}
    for _, input in sim_config["inputs"].items():
        if input["input_type"] == "spikes" and input["module"] == "synapse_replay":
            node_pop, f_name = input["source"], input["spike_file"]
            if not os.path.isabs(f_name):
                warnings.warn("Spike file used for replay is a relative one,"
                              "which will probably break the code when trying to load it")
            spikef_names[node_pop] = f_name
    return spikef_names


def get_proj_spikes(sim_config, t_start, t_end):
    """Loads in input spikes (on projections) using the `bluepysnap.Simulation.config` object"""
    spikef_names = _get_spikef_names(sim_config)
    spikes = {}
    for node_pop, f_name in spikef_names.items():
        tmp = np.loadtxt(f_name, skiprows=1)
        spike_times, spiking_gids = tmp[:, 0], tmp[:, 1].astype(int)
        idx = np.where((t_start < spike_times) & (spike_times < t_end))[0]
        # -1 because spike replay still has an offset in `py-neurodamus`... get rid of it once that's fixed
        spikes[node_pop] = {"spike_times": spike_times[idx], "spiking_gids": spiking_gids[idx] - 1}
    return spikes


# copy-pasted from bglibpy/ssim.py (until BGLibPy will support adding spikes from SpikeFile!)
def _parse_outdat(f_name):
    """Parse the replay spiketrains in a out.dat formatted file"""
    from bluepy.impl.spike_report import SpikeReport  # TODO: fix this
    spikes = SpikeReport.load(f_name).get()
    # convert Series to DataFrame with 2 columns for `groupby` operation
    spike_df = spikes.to_frame().reset_index()
    if (spike_df["t"] < 0).any():
        warnings.warn("Found negative spike times... Clipping them to 0")
        spike_df["t"].clip(lower=0., inplace=True)
    outdat = spike_df.groupby("gid")["t"].apply(np.array)
    return outdat.to_dict()


def get_tc_spikes_bglibpy(sim_config):  # TODO: test this!
    """Loads in input spikes (on projections) using the bluepy.Simulation.config object.
    Returns the format used in BGLibPy spike replay"""
    spikef_names = _get_spikef_names(sim_config)
    spikes = {}
    for _, f_name in spikef_names.items():
        spikes.update(_parse_outdat(f_name))
    return spikes


def group_clusters_by_patterns(clusters, t_bins, stim_times, patterns):
    """Groups clustered sign. activity based on the patterns presented"""
    # get basic info (passing them would be difficult...) and initialize empty matrices
    pattern_names, counts = np.unique(patterns, return_counts=True)
    isi, bin_size = np.max(np.diff(stim_times)), np.min(np.diff(t_bins))
    pattern_matrices = {pattern: np.full((np.max(counts), int(np.ceil(isi / bin_size))), np.nan)
                        for pattern in pattern_names}
    # group sign. activity clusters based on patterns
    row_idx = {pattern: 0 for pattern in pattern_names}
    for pattern, t_start, t_end in zip(patterns, stim_times[:-1], stim_times[1:]):
        idx = np.where((t_start <= t_bins) & (t_bins < t_end))[0]
        if len(idx):
            t_idx = ((t_bins[idx] - t_start) / bin_size).astype(int)
            pattern_matrices[pattern][row_idx[pattern], t_idx] = clusters[idx]
        row_idx[pattern] += 1
    # find max length of sign. activity and cut all matrices there
    max_tidx = np.max([np.sum(~np.all(np.isnan(pattern_matrix), axis=0))
                       for _, pattern_matrix in pattern_matrices.items()])
    pattern_matrices = {pattern_name: pattern_matrix[:, :max_tidx]
                        for pattern_name, pattern_matrix in pattern_matrices.items()}
    # count nr. of clusters per patterns
    pattern_counts, n_clusters = {}, len(np.unique(clusters))
    for pattern_name, matrix in pattern_matrices.items():
        cluster_idx, cluster_counts = np.unique(matrix[~np.isnan(matrix)], return_counts=True)
        counts = np.zeros(n_clusters, dtype=int)
        for i in range(n_clusters):
            if i in cluster_idx:
                counts[i] = cluster_counts[cluster_idx == i]
        pattern_counts[pattern_name] = counts
    return bin_size * max_tidx, row_idx, pattern_matrices, pattern_counts


def count_clusters_by_patterns_across_seeds(all_clusters, t_bins, stim_times, patterns, n_clusters):
    """Counts consensus assemblies across seeds based on the patterns presented"""
    count_matrices = {pattern: np.zeros((len(all_clusters), n_clusters+1), dtype=int) for pattern in np.unique(patterns)}
    seeds = []
    for i, (seed, clusters) in enumerate(all_clusters.items()):
        seeds.append(seed)
        _, _, pattern_matrices, _ = group_clusters_by_patterns(clusters, t_bins[seed], stim_times, patterns)
        for pattern, matrix in pattern_matrices.items():
            cons_assembly_idx, counts = np.unique(matrix, return_counts=True)
            mask = ~np.isnan(cons_assembly_idx)
            for cons_assembly_id, count in zip(cons_assembly_idx[mask], counts[mask]):
                count_matrices[pattern][i, int(cons_assembly_id+1)] = count
    return count_matrices, seeds, np.array([-1] + [i for i in range(n_clusters)])


def load_pkl_df(pklf_name):
    return pd.read_pickle(pklf_name)


def _il_isin(whom, where, parallel):
    """Sirio's in line np.isin() using joblib as parallel backend"""
    if parallel:
        from joblib import Parallel, delayed
        nproc = os.cpu_count() - 1
        with Parallel(n_jobs=nproc, prefer="threads") as p:
            flt = p(delayed(np.isin)(chunk, where) for chunk in np.array_split(whom, nproc))
        return np.concatenate(flt)
    else:
        return np.isin(whom, where)


def get_syn_idx(edgef_name, pre_node_idx, post_node_idx, parallel=True):
    """Returns syn IDs between `pre_node_idx` and `post_node_idx`
    (~1000x faster than c.connectome.pathway_synapses(pre_gids, post_gids))"""
    edges = EdgeStorage(edgef_name)
    edge_pop = edges.open_population(list(edges.population_names)[0])
    # sonata nodes are 0 based (and the functions expect lists of ints)
    afferents_edges = edge_pop.afferent_edges(post_node_idx.astype(int).tolist())
    afferent_nodes = edge_pop.source_nodes(afferents_edges)
    flt = _il_isin(afferent_nodes, pre_node_idx.astype(int), parallel=parallel)
    return afferents_edges.flatten()[flt]


def get_edge_properties(c, edge_pop, syn_idx, properties):
    return c.edges[edge_pop].properties(syn_idx, properties)


def get_synloc_df(c, syn_idx, edge_pop):
    """Loads in synapse location properties (based on precalculated synapse idx)
    needed for detecting synapse clusters"""
    syn_properties = {"@source_node": "pre_gid", "@target_node": "post_gid", "afferent_section_id": "section_id",
                      "afferent_center_x": "x", "afferent_center_y": "y", "afferent_center_z": "z"}
    loc_df = get_edge_properties(c, edge_pop, syn_idx, list(syn_properties.keys()))
    loc_df.rename(columns=syn_properties, inplace=True)
    return loc_df


def get_gid_synloc_df(c, node_id, edge_pop):
    """Loads in synapse location properties (for all afferents of the `node_id`)
    needed for calculating synapses neighbour distances"""
    return c.edges[edge_pop].afferent_edges(node_id, ["@source_node", "afferent_section_id",
                                            "afferent_segment_id", "afferent_segment_offset"]).set_index("@source_node")


def get_edgef_name(c, edge_pop):
    """Gets HDF5 edge file name from circuit object"""
    for tmp in c.config["networks"]["edges"]:
        if list(tmp["populations"].keys())[0] == edge_pop:
            return tmp["edges_file"]
    raise ValueError("Edge population: %s not found in circuit config" % edge_pop)


def get_rho0s(c, node_pop, target, edge_pop):
    """Get initial efficacies (rho0_GB in the SONATA file) for all synapses in the `target`"""
    gids = get_node_idx(c, node_pop, target)
    syn_idx = get_syn_idx(get_edgef_name(c, edge_pop), gids, gids)
    syn_df = get_edge_properties(c, edge_pop, syn_idx, ["@source_node", "@target_node", "rho0_GB"])
    syn_df.rename(columns={"@source_node": "pre_gid", "@target_node": "post_gid", "rho0_GB": "rho"}, inplace=True)
    return syn_df


def save_syn_clusters(save_dir_root, assembly_idx, cluster_df, cross_assembly=False):
    """Saves `cluster_df` with synapse clusters for given assembly"""
    save_dir = os.path.join(save_dir_root, "seed%i" % assembly_idx[1])
    ensure_dir(save_dir)
    if not cross_assembly:
        pklf_name = os.path.join(save_dir, "assembly%i.pkl" % assembly_idx[0])
    else:
        pklf_name = os.path.join(save_dir, "cross_assembly%i.pkl" % (assembly_idx[0]))
    cluster_df.sort_index(inplace=True)
    cluster_df.to_pickle(pklf_name)


def read_base_h5_metadata(h5f_name):
    """Reads 'base' metadata from h5 attributes (root_path, seeds etc.)"""
    h5f = h5py.File(h5f_name, "r")
    return dict(h5f["spikes"].attrs)


def _read_h5_metadata(h5f, group_name=None, prefix=None):
    """Reads metadata from h5 attributes"""
    if prefix is None:
        prefix = "assemblies"
    prefix_grp = h5f[prefix]
    metadata = dict(prefix_grp.attrs)
    if group_name is not None:
        assert group_name in prefix_grp
        metadata.update(dict(prefix_grp[group_name].attrs))
    return metadata


def load_spikes_from_h5(h5f_name, prefix="spikes"):
    """Load spike matrices over seeds from saved h5 file"""
    h5f = h5py.File(h5f_name, "r")
    seeds = list(h5f[prefix].keys())
    project_metadata = _read_h5_metadata(h5f, prefix=prefix)
    prefix_grp = h5f[prefix]
    spike_matrix_dict = {}
    for seed in seeds:
        spike_matrix_dict[seed] = SpikeMatrixResult(prefix_grp[seed]["spike_matrix"][:],
                                                    prefix_grp[seed]["gids"][:],
                                                    prefix_grp[seed]["t_bins"][:])
    h5f.close()
    return spike_matrix_dict, project_metadata


def load_assemblies_from_h5(h5f_name, prefix="assemblies"):
    """Load assemblies over seeds from saved h5 file into dict of AssemblyGroups"""
    from assemblyfire.assemblies import AssemblyGroup
    h5f = h5py.File(h5f_name, "r")
    seeds = list(h5f[prefix].keys())
    project_metadata = {seed: _read_h5_metadata(h5f, seed, prefix) for seed in seeds}
    h5f.close()
    assembly_grp_dict = {seed: AssemblyGroup.from_h5(h5f_name, seed, prefix=prefix) for seed in seeds}
    return assembly_grp_dict, project_metadata


def assembly_groupdic2assembly_grp(assembly_grp_dict):
    from assemblyfire.assemblies import AssemblyGroup
    """Builds 1 big assembly group from assemblies in `assembly_grp_dict` for consensus clustering"""
    gids, n_assemblies, assembly_lst = [], [], []
    for seed, assembly_grp in assembly_grp_dict.items():
        gids.extend(assembly_grp.all.tolist())
        n = len(assembly_grp.assemblies)
        n_assemblies.append(n)
        assembly_lst.extend([assembly_grp.assemblies[i] for i in range(n)])
    return AssemblyGroup(assembly_lst, np.unique(gids), label="all"), n_assemblies


def load_consensus_assemblies_from_h5(h5f_name, prefix="consensus"):
    """Load consensus (clustered and thresholded )assemblies
    from saved h5 file into dict of ConsensusAssembly objects"""
    from assemblyfire.assemblies import ConsensusAssembly
    with h5py.File(h5f_name, "r") as h5f:
        keys = list(h5f[prefix].keys())
    return {k: ConsensusAssembly.from_h5(h5f_name, k, prefix=prefix) for k in keys}


def consensus_dict2assembly_grp(consensus_assemblies):
    """Create AssemblyGroup (object) from dictionary of consensus assemblies
    (AssemblyGroups are used by several functions investigating connectivity to iterate over assemblies...)"""
    from assemblyfire.assemblies import AssemblyGroup
    cons_assembly_idx = np.sort([int(key.split("cluster")[1]) for key in list(consensus_assemblies.keys())])
    all_gids, assembly_lst = [], []
    for cons_assembly_id in cons_assembly_idx:
        cons_assembly = consensus_assemblies["cluster%i" % cons_assembly_id]
        all_gids.extend(cons_assembly.union.gids)
        cons_assembly.idx = (cons_assembly_id, "consensus")
        assembly_lst.append(cons_assembly)
    return AssemblyGroup(assemblies=assembly_lst, all_gids=np.unique(all_gids), label="ConsensusGroup")


def load_syn_nnd_from_h5(h5f_name, n_assemblies, prefix):
    """Loads synapse nearest neighbour results from h5 file
    pd.read_hdf() doesn't understand the structure, so we need to create an object, and access the DataFrame..."""
    from assemblyfire.syn_nnd import SynNNDResults
    with h5py.File(h5f_name, "r") as h5f:
        h5_keys = list(h5f.keys())
    assert prefix in h5_keys, "Prefix not found in HDF5 file"
    results = SynNNDResults(h5f_name, n_assemblies, prefix)
    df = results._df.copy()  # TODO: fix the access in the class
    df.set_index(("gid", "gid"), inplace=True)
    df.index = df.index.astype(int)  # studpid pandas
    df.index.name = "gid"  # studpid pandas
    return df.sort_index()


def load_single_cell_features_from_h5(h5f_name, prefix="single_cell"):
    """Load spike matrices over seeds from saved h5 file"""
    h5f = h5py.File(h5f_name, "r")
    prefix_grp = h5f[prefix]
    single_cell_features = {"gids": prefix_grp["gids"][:], "r_spikes": prefix_grp["r_spikes"][:]}
    h5f.close()
    return single_cell_features


def read_cluster_seq_data(h5f_name):
    """Load metadata needed (stored under diff. prefixes) for re-plotting cluster (of time bin) sequences"""
    h5f = h5py.File(h5f_name, "r")
    spikes_metadata = _read_h5_metadata(h5f, prefix="spikes")
    seeds = ["seed%i" % seed for seed in spikes_metadata["seeds"]]
    assemblies_metadata = {seed: _read_h5_metadata(h5f, seed, "assemblies") for seed in seeds}
    metadata = {"clusters": {seed: assemblies_metadata[seed]["clusters"] for seed in seeds},
                "t_bins": {seed: h5f["spikes"][seed]["t_bins"][:] for seed in seeds},
                "stim_times": spikes_metadata["stim_times"],
                "patterns": spikes_metadata["patterns"]}
    h5f.close()
    return metadata
