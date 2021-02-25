from assemblyfire import Assembly, AssemblyGroup, ConsensusAssembly
import numpy

all_gids = list(range(500))
rnd_gids_a = [numpy.unique(numpy.random.randint(0, 500, 75)) for _ in range(3)]
rnd_gids_b = [numpy.unique(numpy.random.randint(0, 500, 75)) for _ in range(5)]

assemblies_a = [Assembly(gids, index=(0, i)) for i, gids in enumerate(rnd_gids_b)]
assemblies_a2 = [Assembly(gids, index=i) if i > 0 else Assembly(gids, index=None)
                 for i, gids in enumerate(rnd_gids_b)]
assemblies_b = [Assembly(gids, index=(0, i)) for i, gids in enumerate(rnd_gids_b)]

asmbl_grp = AssemblyGroup(assemblies_a, all_gids, label="Group", metadata=None)
asmbl_grp2 = AssemblyGroup(assemblies_a2, all_gids, label="Group2", metadata=None)
cons_asmbl = ConsensusAssembly(assemblies_b, (1, 0), label="Consensus")

asmbl_grp.to_h5("test_assembly_io.h5", prefix="a_group")
asmbl_grp2.to_h5("test_assembly_io.h5", prefix="a_group")
cons_asmbl.to_h5("test_assembly_io.h5", prefix="a_consensus")

asmbl_grp_read = AssemblyGroup.from_h5("test_assembly_io.h5", "Group", prefix="a_group")
asmbl_grp2_read = AssemblyGroup.from_h5("test_assembly_io.h5", "Group2", prefix="a_group")
cons_asmbl_read = ConsensusAssembly.from_h5("test_assembly_io.h5", "Consensus", prefix="a_consensus")

print("BEFORE:")
print(cons_asmbl.gids)
print(cons_asmbl.idx)

print("AFTER:")
print(cons_asmbl_read.gids)
print(cons_asmbl_read.idx)

for bf, af in zip(asmbl_grp, asmbl_grp_read):
    print("BEFORE:")
    print(bf.gids)
    print(bf.idx)

    print("AFTER:")
    print(af.gids)
    print(af.idx)

for bf, af in zip(asmbl_grp2, asmbl_grp2_read):
    print("BEFORE:")
    print(bf.gids)
    print(bf.idx)

    print("AFTER:")
    print(af.gids)
    print(af.idx)