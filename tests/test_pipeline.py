"""
Testing the analysis pipeline on synthetic spike trains
last modified: András Ecker 06.2023
"""

from assemblyfire.utils import ensure_dir
from assemblyfire.spikes import SpikeMatrixGroup
from assemblyfire.clustering import cluster_spikes, detect_assemblies


if __name__ == "__main__":
    spikes = SpikeMatrixGroup("test.yaml")
    ensure_dir(spikes.fig_path)

    spike_matrix_dict, project_metadata = spikes.get_sign_spike_matrices()
    clusters_dict = cluster_spikes(spike_matrix_dict, {}, project_metadata, spikes.fig_path)
    detect_assemblies(spike_matrix_dict, clusters_dict, spikes.core_cell_th_pct, spikes.h5f_name,
                      spikes.h5_prefix_assemblies, None, spikes.fig_path)

