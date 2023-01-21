"""
CLI to detect and analyse cell assemblies
 * `assemblyfire assemblies config_path`: detect assemblies in spiking data
 * `assemblyfire consensus config_path`: create consensus assemblies from assemblies across seeds
 * `assemblyfire conn-mat config_path`: gets connectivity matrix
 * `assemblyfire syn-clust config_path`: finds clusters of synapses in assembly neurons
 * `assemblyfire syn-nnd config_path assembly_group_name`: gets synapses nearest neighbour distances
 * `assemblyfire rerun config_path seed gid`: reruns single gid in BGLibPy (with extra reporting and modifications)
 * `assemblyfire single-cell config_path`: gets single cell features from simulations
last modified: Thomas Delemontex, András Ecker 01.2023
"""

import click
import logging

logging.basicConfig(level=logging.INFO)
L = logging.getLogger("assemblyfire")


def set_verbose(logger, verbose):
    """Set the verbose level for the CLI"""
    logger.setLevel((logging.WARNING, logging.INFO, logging.DEBUG)[min(verbose, 2)])


@click.group()
@click.option('-v', '--verbose', count=True)
def cli(verbose):
    """CLI entry point."""
    set_verbose(L, verbose)


@cli.command()
@click.argument("config_path", required=True)
def assemblies(config_path):
    """CLI for `find_assemblies.py/run()`"""
    from assemblyfire.find_assemblies import run
    run(config_path)
    

@cli.command()
@click.argument("config_path", required=True)
def consensus(config_path):
    """CLI for `find_consensus_assemblies.py/run()`"""
    from assemblyfire.find_consensus_assemblies import run
    run(config_path)


@cli.command()
@click.argument("config_path", required=True)
def conn_mat(config_path):
    """CLI for `get_connectivity_matrix.py/run()`"""
    from assemblyfire.get_connectivity_matrix import run
    run(config_path)


@cli.command()
@click.argument("config_path", required=True)
@click.argument("debug", required=False, default=False)
def syn_clust(config_path, debug):
    """CLI for `find_synapse_clusters.py/run()`"""
    from assemblyfire.find_synapse_clusters import run
    run(config_path, debug)


@cli.command()
@click.argument("config_path", required=True)
@click.argument("assembly_grp_name", required=True)
@click.argument("buf_size", required=False, default=100)
@click.argument("seed", required=False, default=100)
def syn_nnd(config_path, assembly_grp_name, buf_size, seed):
    """CLI for `get_synapse_nnds.py/run()`"""
    from assemblyfire.get_synapse_nnds import run
    run(config_path, assembly_grp_name, buf_size, seed)


@cli.command()
@click.argument("config_path", required=True)
@click.argument("seed", required=True)
@click.argument("gid", required=True)
def rerun(config_path, seed, gid):
    """CLI for `rerun_single_cell.py/run()`"""
    from assemblyfire.rerun_single_cell import run
    run(config_path, seed, gid)


@cli.command()
@click.argument("config_path", required=True)
def single_cell(config_path):
    """CLI for `get_single_cell_features.py/run()`"""
    from assemblyfire.get_single_cell_features import run
    run(config_path)


@cli.command()
@click.argument("config_path", required=True)
@click.argument("prefix", required=True)
def clean_h5(config_path, prefix):
    """Removes data under the given `prefix` in the HDF5 file"""
    import h5py
    from assemblyfire.config import Config
    config = Config(config_path)
    with h5py.File(config.h5f_name, "a") as h5f:
        assert prefix in list(h5f.keys())
        del h5f[prefix]
