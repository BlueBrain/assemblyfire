"""
Functionalities to investigate the relationship between synaptc clustering and dendritic nonlinearities
1st step: writes launch scripts that rerun selected cells in BGLibPy (see `assemblyfire/rerun_single_cell.py`)
          (with extra reporting and some modifications)
last modified: András Ecker 01.2023
"""

import os
import h5py

import assemblyfire.utils as utils
from assemblyfire.config import Config


DSET_MEMBER = "member"
DSET_DEG = "degree"
DSET_CLST = "strength"
DSET_PVALUE = "pvalue"

slurm_template = """\
#!/bin/sh
#SBATCH --job-name={job_name}
#SBATCH --account=proj96
#SBATCH --partition=prod
#SBATCH --constraint=cpu
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=2
#SBATCH --time=6:00:00
#SBATCH --mem=15g
#SBATCH --no-requeue
#SBATCH --output={log_name}

source /gpfs/bbp.cscs.ch/project/proj96/home/ecker/assemblyfire/setupenv.sh
assemblyfire -v rerun {config_path} {seed} {gid}\n
"""


def write_launchscripts(config_path, n_gids=10, p_th=0.05):
    """Loads synapse nearest neighbour distance results for each seed (that has any saved),
    in each assembly finds the top `n_gids` with the highest indegree and significant nnd. 'strength',
    and writes launchscripts for them"""
    config = Config(config_path)
    assembly_grp_dict, _ = utils.load_assemblies_from_h5(config.h5f_name, config.h5_prefix_assemblies)

    sbatch_dir = os.path.join(config.root_path, "sbatch")
    utils.ensure_dir(sbatch_dir)
    with h5py.File(config.h5f_name, "r") as h5f:
        h5_keys = list(h5f.keys())
    simulated_gids = {}

    for seed, assembly_grp in assembly_grp_dict.items():
        prefix = "%s_syn_nnd" % seed
        if prefix in h5_keys:  # since this runs forever one might not have the results for all seeds
            syn_nnds = utils.load_syn_nnd_from_h5(config.h5f_name, len(assembly_grp), prefix=prefix)
            assembly_idx = syn_nnds.columns.get_level_values(0).unique().to_numpy()
            f_names, simulated_gids[seed] = [], {}
            for assembly_id in assembly_idx:
                df = syn_nnds.loc[:, [(assembly_id, DSET_MEMBER), (assembly_id, DSET_DEG),
                                      (assembly_id, DSET_CLST), (assembly_id, DSET_PVALUE)]]
                df.columns = df.columns.get_level_values(1)
                df[DSET_CLST] *= -1  # TODO: get rid of this when the new results are ready (the new code takes this *-1 into acc.)
                # index out assembly members with significant syn nnd. 'strength'
                df = df.loc[(df[DSET_MEMBER] == 1) & (df[DSET_CLST] > 0) & (df[DSET_PVALUE] < p_th)]
                # get the gids of the first n highest indegrees
                df = df.sort_values(DSET_DEG, ascending=False)
                gids = df.iloc[:n_gids].index.to_numpy()
                simulated_gids[seed][assembly_id] = gids
                for gid in gids:
                    f_name = os.path.join(sbatch_dir, "%s_%s_a%i.batch" % (seed, assembly_id, gid))
                    f_names.append(f_name)
                    with open(f_name, "w+", encoding="latin1") as f:
                        f.write(slurm_template.format(job_name="rerun_a%i" % gid,
                                                      log_name=os.path.splitext(f_name)[0] + ".log",
                                                      config_path=config_path, seed=int(seed.split("seed")[1]), gid=gid))
            # write one big launchscript that launches them all
            with open(os.path.join(sbatch_dir, "launch_%s.sh" % seed), "w") as f:
                for f_name in f_names:
                    f.write("sbatch %s\n" % f_name)


if __name__ == "__main__":
    config_path = "/gpfs/bbp.cscs.ch/project/proj96/home/ecker/assemblyfire/configs/v7_10seeds_np.yaml"
    write_launchscripts(config_path)

