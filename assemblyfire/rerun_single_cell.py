"""
TODO
"""

import os
import logging
import numpy as np
import pandas as pd

import bglibpy

from assemblyfire.config import Config
import assemblyfire.utils as utils

L = logging.getLogger("assemblyfire")


def get_gid_instantiation_vars(ssim):
    """Gets variables necessary to (properly) instantiate a gid in BGLibPy"""
    ca = ssim.circuit_access
    # on has to manually add TC spikes (as BGLibPy doesn't load spikes from the SpikeFile in the BlueConfig)
    proj_spike_trains = utils.get_tc_spikes_bglibpy(ca.config.bc)
    # instead of all the cell's synapses get only the ones that originate from the sim's target
    # and the ones from the active TC fibers (TC fibers don't have minis - so no need to add all TC synapses)
    pre_gids = np.concatenate([ca._bluepy_sim.target_gids, np.array(list(proj_spike_trains.keys()))])
    return pre_gids, proj_spike_trains, ca.config.bc.Run_Default.SpikeLocation


def run_sim(ssim, gid, pre_gids, pre_spike_trains, spike_loc):
    """TODO"""
    # instantiate gid with replay on all synapses and the same input as it gets in the network simulation
    ssim.instantiate_gids([gid], add_synapses=True, add_projections=True, add_minis=True, intersect_pre_gids=pre_gids,
                          add_stimuli=True, add_replay=True, pre_spike_trains=pre_spike_trains)
    cell = ssim.cells[gid]
    all_sections = cell.cell.getCell().all

    # set up fake NetCon to detect spikes
    nc = cell.create_netcon_spikedetector(None, location=spike_loc)
    spike_vec = bglibpy.neuron.h.Vector()
    nc.record(spike_vec)
    # record voltage from all sections (not just the soma...) built in BGLibPy fn. ignore `record_dt`
    for section in all_sections:
        cell.add_recording("neuron.h." + section.name() + "(0.5)._ref_v", dt=ssim.record_dt)

    ssim.run()

    spike_times = np.array(spike_vec)
    # get voltage recordings from all sections
    data, columns = [], []
    for section in all_sections:
        columns.append(section.name().split(".")[1])
        data.append(cell.get_recording("neuron.h." + section.name() + "(0.5)._ref_v").reshape(-1, 1))
    vs = pd.DataFrame(data=np.hstack(data), columns=columns, index=cell.get_time())
    vs.index.name = "time"

    return spike_times[spike_times > 0], vs


def main(config_path, seed, gid=8473):
    """TODO"""

    config = Config(config_path)
    save_dir = os.path.join(config.root_path, "analyses", "allsec_voltages")
    utils.ensure_dir(save_dir)
    sim_path = utils.get_sim_path(config.root_path).loc[seed]
    L.info(" Instantiating %i from %s in BGLibPy" % (gid, sim_path))
    ssim = utils.get_bglibpy_ssim(sim_path)
    pre_gids, pre_spike_trains, spike_loc = get_gid_instantiation_vars(ssim)

    L.info(" Running sim w/ baseline conditions ")
    spike_times, vs = run_sim(ssim, 8473, pre_gids, pre_spike_trains, spike_loc)
    vs.to_pickle(os.path.join(save_dir, "seed%s_a%i_baseline_voltages.pkl" % (seed, gid)))
    ssim.delete()


