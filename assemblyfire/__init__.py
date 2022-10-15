"""assemblyfire"""

from assemblyfire.version import __version__
from assemblyfire.spikes import SpikeMatrixGroup, spikes2mat
from assemblyfire.assemblies import Assembly, AssemblyGroup, ConsensusAssembly
from assemblyfire.topology import AssemblyTopology
from assemblyfire.clustering import cluster_sim_mat, cluster_spikes, detect_assemblies, cluster_assemblies, cluster_synapses
from assemblyfire import utils
from assemblyfire import plots

