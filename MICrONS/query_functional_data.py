"""
Query co-registered functional data from the MICrONS dataset
last modified: András Ecker 09.2023
"""

import numpy as np
import pandas as pd
# follow https://github.com/cajal/microns-nda-access to be able to run this!
from caveclient import CAVEclient
from microns_phase3 import nda


def get_data(matched_df, session_id, scan_id):
    """Get extracted spike traces from given scan and session"""
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
    # `fetch` gets all extracted spikes in one go, but then idk. how to map them to `id_ref`...
    df = matched_df.loc[(matched_df["session"] == session_id) & (matched_df["scan_idx"] == scan_id)]
    spikes = np.zeros((len(df), len(t)), dtype=np.float32)
    for i, unit_id in enumerate(df["unit_id"].to_numpy()):
        unit_key = {"session": session_id, "scan_idx": scan_id, "unit_id": unit_id}
        spikes[i, :] = (nda.Activity & unit_key).fetch1("trace")
    npzf_name = "MICrONS_session%i_scan%i.npz" % (session_id, scan_id)
    np.savez(npzf_name, spikes=spikes, t=t, idx=df["id_ref"].to_numpy(),
             v_treadmill=v_treadmill, pattern_names=pattern_names, stim_times=stim_times)


if __name__ == "__main__":
    client = CAVEclient("minnie65_public")
    client.materialize.version = 661
    matched_df = client.materialize.query_table("coregistration_manual_v3")

    # get data from scans that have many neurons and most of them are co-registered
    gids = np.genfromtxt("id_ref.txt").astype(int)  # saved from structural data
    df = matched_df.groupby(["session", "scan_idx"])["id_ref"].apply(np.array)
    for mi, idx in df.items():
        if len(idx) >= 1000 and np.in1d(idx, gids).sum() / len(idx) * 100 >= 85:
            session_id, scan_id = mi[0], mi[1]
            get_data(matched_df, session_id, scan_id)


