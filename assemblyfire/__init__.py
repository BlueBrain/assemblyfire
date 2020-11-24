"""assemblyfire"""

from assemblyfire.version import __version__
from assemblyfire.spikes import SpikeMatrixGroup
from assemblyfire.assemblies import AssemblyProjectMetadata, Assembly, AssemblyGroup, ConsensusAssembly
from assemblyfire.clustering import cluster_spikes, detect_assemblies, cluster_assemblies
from assemblyfire.utils import load_assemblies_from_h5
