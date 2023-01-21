"""assemblyfire"""

from assemblyfire.version import __version__
from assemblyfire.spikes import SpikeMatrixGroup, spikes2mat
from assemblyfire.assemblies import Assembly, AssemblyGroup, ConsensusAssembly
from assemblyfire.topology import AssemblyTopology
from assemblyfire.syn_nnd import SynNNDResults
from assemblyfire.clustering import cluster_sim_mat, cluster_spikes, get_core_cell_idx, detect_assemblies,\
                                    cluster_assemblies, syn_nearest_neighbour_distances, cluster_synapses
from assemblyfire import utils
from assemblyfire import plots

