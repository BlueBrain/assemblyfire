"""
Config file class
authors: Thomas Delemontex and András Ecker; last update 04.2023
"""

import os
import yaml


class Config(object):
    """Class to store config parameters about the simulations and analysis"""

    def __init__(self, config_path):
        """YAML config file based constructor"""
        self._config_path = config_path
        with open(config_path, "r") as f:
            self._config = yaml.load(f, Loader=yaml.SafeLoader)

    @property
    def config(self):
        return self._config

    @property
    def root_path(self):
        return self.config["root_path"]

    @property
    def input_sequence_fname(self):
        f_name = self.config["input_sequence_fname"]
        return f_name if os.path.isabs(f_name) else os.path.join(self.root_path, f_name)

    @property
    def pattern_nodes_fname(self):
        f_name = self.config["pattern_nodes_fname"]
        return f_name if os.path.isabs(f_name) else os.path.join(self.root_path, f_name)

    @property
    def patterns_edges(self):
        return self.config["patterns_edges"] if "patterns_edges" in self.config else {}

    @property
    def pattern_locs_fname(self):
        if len(self.patterns_edges):
            return os.path.join(os.path.split(self.pattern_nodes_fname)[0],
                                "%s.txt" % list(self.patterns_edges.keys())[0])
        else:
            return None

    @property
    def h5f_name(self):
        f_name = self.config["h5_out"]["file_name"]
        return f_name if os.path.isabs(f_name) else os.path.join(self.root_path, f_name)

    @property
    def h5_prefixes(self):
        return self.config["h5_out"]["prefixes"]

    @property
    def h5_prefix_spikes(self):
        return self.config["h5_out"]["prefixes"]["spikes"]

    @property
    def h5_prefix_avg_spikes(self):
        return self.config["h5_out"]["prefixes"]["average_spikes"]

    @property
    def h5_prefix_assemblies(self):
        return self.config["h5_out"]["prefixes"]["assemblies"]

    @property
    def h5_prefix_consensus_assemblies(self):
        return self.config["h5_out"]["prefixes"]["consensus_assemblies"]

    @property
    def h5_prefix_avg_assemblies(self):
        return self.config["h5_out"]["prefixes"]["average_assemblies"]

    @property
    def h5_prefix_connectivity(self):
        return self.config["h5_out"]["prefixes"]["connectivity"]

    @property
    def h5_prefix_single_cell(self):
        return self.config["h5_out"]["prefixes"]["single_cell_features"]

    @property
    def root_fig_path(self):
        return self.config["root_fig_path"]

    @property
    def fig_path(self):
        h5f_name = self.config["h5_out"]["file_name"].split('.')[0]
        if h5f_name == "assemblies":
            return os.path.join(self.root_fig_path, os.path.split(self.root_path)[1])
        else:
            return os.path.join(self.root_fig_path, os.path.split(self.root_path)[1] + h5f_name.split("assemblies")[1])

    @property
    def node_pop(self):
        return self.config["preprocessing_protocol"]["node_pop"]\
            if "node_pop" in self.config["preprocessing_protocol"] else None

    @property
    def edge_pop(self):
        return self.config["edge_pop"] if "edge_pop" in self.config else None

    @property
    def target(self):
        return self.config["preprocessing_protocol"]["target"]

    @property
    def t_start(self):
        return self.config["preprocessing_protocol"]["t_start"]

    @property
    def t_end(self):
        return self.config["preprocessing_protocol"]["t_end"]

    @property
    def t_chunks(self):
        return self.config["preprocessing_protocol"]["t_chunks"]\
            if "t_chunks" in self.config["preprocessing_protocol"] else None

    @property
    def bin_size(self):
        return self.config["preprocessing_protocol"]["bin_size"]

    @property
    def threshold_rate(self):
        return self.config["preprocessing_protocol"]["threshold_rate"] \
            if "threshold_rate" in self.config["preprocessing_protocol"] else True

    @property
    def surr_rate_method(self):
        return self.config["preprocessing_protocol"]["surr_rate_method"] \
            if "surr_rate_method" in self.config["preprocessing_protocol"] else "Sasaki"

    @property
    def ignore_seeds(self):
        return self.config["preprocessing_protocol"]["ignore_seeds"] \
            if "ignore_seeds" in self.config["preprocessing_protocol"] else []

    @property
    def core_cell_th_pct(self):
        core_cell_th_pct = 95  # default 95% percentile from Herzog et al. 2021
        if "clustering" in self.config:
            if "core_cell_th_pct" in self.config["clustering"]:
                core_cell_th_pct = self.config["clustering"]["core_cell_th_pct"]
        return core_cell_th_pct

    @property
    def overwrite_seeds(self):
        overwrite_seeds = {}
        if "clustering" in self.config:
            if "overwrite_n_clusters" in self.config["clustering"]:
                overwrite_seeds = self.config["clustering"]["overwrite_n_clusters"]
        return overwrite_seeds

    @property
    def syn_clustering_target_range(self):
        return self.config["clustering"]["synapses"]["target_range"]

    @property
    def syn_clustering_min_nsyns(self):
        return self.config["clustering"]["synapses"]["min_nsyns"]

    @property
    def syn_clustering_mtypes(self):
        return self.config["clustering"]["synapses"]["mtypes"]\
                if "mtypes" in self.config["clustering"]["synapses"] else ["L5_TPC:A", "L5_TPC:B"]

    @property
    def syn_clustering_n_neurons_sample(self):
        return self.config["clustering"]["synapses"]["n_neurons_sample"]

    @property
    def syn_clustering_save_dir(self):
        h5f_name = self.config["h5_out"]["file_name"].split('.')[0]
        if h5f_name == "assemblies":
            return os.path.join(self.root_path, "analyses", "syn_clusters")
        else:
            return os.path.join(self.root_path, "analyses", "syn_clusters" + h5f_name.split("assemblies")[1])

    @property
    def syn_clustering_cross_assemblies(self):
        return self.config["clustering"]["synapses"]["cross_assemblies"]\
                if "cross_assemblies" in self.config["clustering"]["synapses"] else {}

    @property
    def manuscript_assembly_sequences(self):
        return self.config["manuscript_figures"]["assembly_sequences"]\
                if "manuscript_figures" in self.config else {}
