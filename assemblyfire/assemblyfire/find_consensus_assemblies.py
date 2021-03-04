# -*- coding: utf-8 -*-
"""
Main run function for finding consensus cell assemblies
last modified: András Ecker 11.2020
"""

import os
import yaml
import logging
from cached_property import cached_property

from assemblyfire.utils import load_assemblies_from_h5

L = logging.getLogger("assemblyfire")


class ConsensusConfig(object):
    """Class to store config parameters about consensus assemblies"""

    def __init__(self, config_path):
        """YAML config file based constructor"""
        self._config_path = config_path
        with open(config_path, "r") as f:
            self._config = yaml.load(f, Loader=yaml.SafeLoader)

    @property
    def config(self):
        return self._config

    @property
    def h5f_name(self):
        return self.config["h5_out"]["file_name"]

    @property
    def h5_prefix_assemblies(self):
        return self.config["h5_out"]["prefixes"]["assemblies"]

    @property
    def h5_prefix_consensus(self):
        return self.config["h5_out"]["prefixes"]["consensus_assemblies"]

    @property
    def root_fig_path(self):
        return self.config["root_fig_path"]

    @cached_property
    def fig_path(self):
        return os.path.join(self.root_fig_path, self._config_path.split('/')[-1][:-5])

    @property
    def clustering_method(self):
        assert self.config["clustering_methods"]["assemblies"] in ["hamming", "greedy"]
        return self.config["clustering_methods"]["assemblies"]


def run(config_path):
    """
    Loads in project related info from yaml config file ...
    :param config_path: str - path to project config file
    """

    config = ConsensusConfig(config_path)
    L.info(" Load in assemblies from %s" % config.h5f_name)
    assembly_grp_dict = load_assemblies_from_h5(config.h5f_name, config.h5_prefix_assemblies, load_metadata=False)

    L.info(" Creating consensus assemblies and saving them to the same file")
    if config.clustering_method == "hierarchical":
        from assemblyfire.assemblies import consensus_over_seeds_hc
        consensus_over_seeds_hc(assembly_grp_dict, config.h5f_name, config.h5_prefix_consensus, config.fig_path)
    elif config.clustering_method == "greedy":  # TODO: make this run
        from assemblyfire.assemblies import consensus_over_seeds_greedy
        consensus_assemblies, _, _ = consensus_over_seeds_greedy(assembly_grp_dict)




