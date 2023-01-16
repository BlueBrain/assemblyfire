import pyflagsercount
import numpy
import pandas
from assemblyfire.topology import AssemblyTopology
from assemblyfire.assemblies import AssemblyGroup
from tqdm import tqdm
from assemblyfire import utils


def consensus_assembly_group_for(config):
    assembly_grp_dict = utils.load_consensus_assemblies_from_h5(config.h5f_name, config.h5_prefix_consensus_assemblies)
    all_gids = []
    for cons in assembly_grp_dict.values():
        all_gids = numpy.union1d(all_gids, cons.union.gids)

    kk = sorted(assembly_grp_dict.keys())
    for _k in kk:
        assembly_grp_dict[_k].idx = (assembly_grp_dict[_k].idx, "consensus")
        
    cons_grp = AssemblyGroup([assembly_grp_dict[_k] for _k in kk], all_gids, label="ConsensusGroup")
    return cons_grp


def higher_dim_indegree(config, seeds=None):
    conn_mat = AssemblyTopology.from_h5(config.h5f_name,
                                    prefix=config.h5_prefix_connectivity, group_name="full_matrix")

    assembly_grp_dict, _ = utils.load_assemblies_from_h5(config.h5f_name, config.h5_prefix_assemblies)
    if seeds is None: seeds = list(assembly_grp_dict.items())
    if "consensus" in seeds:
        assembly_grp_dict["consensus"] = consensus_assembly_group_for(config)

    gids = conn_mat.gids
    dims = [2, 3, 4, 5, 6]
    res = {}

    for seed in seeds:
        assembly_grp = assembly_grp_dict[seed]
        for assembly in tqdm(assembly_grp.assemblies, desc="Iterating over assemblies"):
            simplex_list = conn_mat.simplex_list(assembly.gids, gids)
            in_cons = conn_mat.submatrix(assembly.gids, sub_gids_post=gids).tocoo().col
            sink_counts, _ = numpy.histogram(in_cons, numpy.arange(len(gids) + 1))
            res[(seed, "assembly{0}".format(assembly.idx[0]), "degree", "dim1")] = sink_counts
            for dim in dims:
                if dim < len(simplex_list):
                    sink_counts, _ = numpy.histogram(simplex_list[dim][:, -1], numpy.arange(len(gids) + 1))
                else:
                    sink_counts = numpy.zeros(len(gids), dtype=int)
                res[(seed, "assembly{0}".format(assembly.idx[0]), "degree", "dim{0}".format(dim))] = sink_counts

    res[("gid", "gid", "gid", "gid")] = gids
    df = pandas.DataFrame(res)
    return df

if __name__ == "__main__":
    import sys
    from assemblyfire.config import Config

    fn_out = sys.argv[1]
    if len(sys.argv) > 2:
        seeds = [sys.argv[2]]
    else:
        seeds = None

    config = Config("../configs/v7_10seeds_np.yaml")
    DF = higher_dim_indegree(config, seeds=seeds)
    DF.to_hdf(fn_out, "indegree_by_dimension")
