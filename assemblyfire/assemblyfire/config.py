# -*- coding: utf-8 -*-
"""
Config file class
authors: Thomas Delemontex and András Ecker; last update 11.2021
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
    def patterns_fname(self):
        f_name = self.config["patterns_fname"]
        return f_name if os.path.isabs(f_name) else os.path.join(self.root_path, f_name)

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
    def h5_prefix_assemblies(self):
        return self.config["h5_out"]["prefixes"]["assemblies"]

    @property
    def h5_prefix_consensus_assemblies(self):
        return self.config["h5_out"]["prefixes"]["consensus_assemblies"]

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
        return os.path.join(self.root_fig_path, os.path.split(self.root_path)[1])

    @property
    def target(self):
        return self.config["preprocessing_protocol"]["target"]

    @property
    def t_start(self):
        return self.config["preprocessing_protocol"]["t_start"]

    @property
    def t_start_exception(self):
        if "t_start_exception" in self.config["preprocessing_protocol"]:
            return self.config["preprocessing_protocol"]["t_start_exception"]
        else:
            return self.t_start

    @property
    def seeds_exception(self):
        if "seeds_exception" in self.config["preprocessing_protocol"]:
            return self.config["preprocessing_protocol"]["seeds_exception"]
        else:
            return []

    @property
    def ignore_seeds(self):
        if "ignore_seeds" in self.config["preprocessing_protocol"]:
            return self.config["preprocessing_protocol"]["ignore_seeds"]
        else:
            return []

    @property
    def t_end(self):
        return self.config["preprocessing_protocol"]["t_end"]

    @property
    def bin_size(self):
        return self.config["preprocessing_protocol"]["bin_size"]

    @property
    def threshold_rate(self):
        return self.config["preprocessing_protocol"]["threshold_rate"]

    @property
    def spike_clustering_method(self):
        assert self.config["clustering"]["spikes"]["method"] in ["hierarchical", "density_based"]
        return self.config["clustering"]["spikes"]["method"]

    @property
    def overwrite_seeds(self):
        if "overwrite_n_clusters" in self.config["clustering"]["spikes"]:
            return self.config["clustering"]["spikes"]["overwrite_n_clusters"]
        else:
            return {}

    @property
    def assembly_clustering_method(self):
        assert self.config["clustering"]["assemblies"]["method"] in ["hierarchical", "greedy"]
        return self.config["clustering"]["assemblies"]["method"]

