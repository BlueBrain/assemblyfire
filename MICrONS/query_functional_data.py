"""
Query co-registered functional data from the MICrONS dataset
last modified: András Ecker 09.2023
"""

import numpy as np
import pandas as pd
# follow https://github.com/cajal/microns-nda-access to be able to run this!
from caveclient import CAVEclient
from microns_phase3 import nda


def get_data(matched_df):
    """Get extracted spike traces from the last scan of the last session"""
    session_id = matched_df["session"].max()
    df = matched_df.loc[matched_df["session"] == session_id]
    scan_id = df["scan_idx"].max()  # could loop over these to build consensus assemblies...
    df = df.loc[df["scan_idx"] == scan_id]
    unit_key = {"session": session_id, "scan_idx": scan_id}
    nframes, fps = (nda.Scan & unit_key).fetch1("nframes", "fps")
    t = np.arange(nframes) / fps
    # r_pupil = (nda.ManualPupil & unit_key).fetch1("pupil_maj_r")  # always return nans...
    v_treadmill = (nda.Treadmill & unit_key).fetch1("treadmill_velocity")
    trial_idx, pattern_names, stim_times = (nda.Trial & unit_key).fetch("trial_idx", "type", "start_frame_time")
    pattern_names = np.array([pattern_name.split('.')[1] for pattern_name in pattern_names])
    # get a bit more info about the type of "Clip"
    for trial_id in trial_idx[pattern_names == "Clip"]:
        trial_key = {"session": session_id, "scan_idx": scan_id, "trial_idx": trial_id}
        pattern_names[trial_id] = "Clip/%s" % ((nda.Trial & trial_key) * nda.Clip).fetch1("short_movie_name").upper()
    # `fetch` gets all extracted spikes in one go, but then idk. how to map them to `pt_root_id`...
    spikes = np.zeros((len(df), len(t)), dtype=np.float32)
    for i, unit_id in enumerate(df["unit_id"].to_numpy()):
        unit_key = {"session": session_id, "scan_idx": scan_id, "unit_id": unit_id}
        spikes[i, :] = (nda.Activity & unit_key).fetch1("trace")
    npzf_name = "MICrONS_session%i_scan%i.npz" % (session_id, scan_id)
    np.savez(npzf_name, spikes=spikes, t=t, idx=df["pt_root_id"].to_numpy(),
             v_treadmill=v_treadmill, pattern_names=pattern_names, stim_times=stim_times)


if __name__ == "__main__":
    client = CAVEclient("minnie65_public")
    client.materialize.version = 661
    matched_df = client.materialize.query_table("coregistration_manual_v3")
    get_data(matched_df)


