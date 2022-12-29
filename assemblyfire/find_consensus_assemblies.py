"""
Main run function for finding consensus assemblies
last modified: András Ecker 10.2022
"""

import logging

from assemblyfire.config import Config
from assemblyfire.utils import load_assemblies_from_h5
from assemblyfire.assemblies import consensus_over_seeds

L = logging.getLogger("assemblyfire")


def run(config_path):
    """
    Loads in project related info from yaml config file, finds consensus assemblies and saves them to h5
    :param config_path: str - path to project config file
    """
    config = Config(config_path)
    L.info(" Load in assemblies from %s" % config.h5f_name)
    assembly_grp_dict, _ = load_assemblies_from_h5(config.h5f_name, config.h5_prefix_assemblies)
    L.info(" Creating consensus assemblies and saving them to the same file")
    consensus_over_seeds(assembly_grp_dict, config.h5f_name, config.h5_prefix_consensus_assemblies, config.fig_path)




