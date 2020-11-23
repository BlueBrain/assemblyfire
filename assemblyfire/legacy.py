# -*- coding: utf-8 -*-
"""
Legacy code which might be usefull to keep...
"""


def load_assemblies_from_h5(fn, prefix=None):
    import h5py
    from assemblyfire import assemblies
    if prefix is None:
        prefix = "assemblies"
    with h5py.File(fn, "r") as h5:
        kk = list(h5[prefix].keys())
    project_metadata = assemblies.AssemblyProjectMetadata.from_h5(fn, prefix=prefix)
    return dict([(k, assemblies.AssemblyGroup.from_h5(fn, k, prefix=prefix))
                 for k in kk]), project_metadata


def project_assemblies_to_h5(dict_groups, fn, metadata=None, prefix=None):
    from assemblyfire import assemblies
    if metadata is None:
        metadata = {}
    assemblies.AssemblyProjectMetadata.to_h5(metadata, fn, prefix=prefix)
    for assembly_group in dict_groups.values():
        assembly_group.to_h5(fn, prefix=prefix)


def consensus_over_seeds(assembly_grp_dict, score_function="correlation", threshold=0.94):
    """
    Generate consensus assemblies the following way:
    1. Perform alignment to a reference AssemblyGroup, using each group once as the reference
    2. Put the results into a (n_seeds * n_assemblies) x (n_seeds * n_assemblies) matrix as follows:
       M[a * n_assemblies + i, b * n_assemblies + j] is a quality assessment of how well assembly i of seed a
       aligns with assembly j of seed b.
    3. Create graph from M, applying a threshold to it. Nodes of the graph represent assemblies
    4. Find connected components in the graph. Assemblies in each component are merged into one consensus assembly
    :param assembly_grp_dict: (dict) containing AssemblyGroups
    :return: (AssemblyGroup) Consensus assemblies
    """
    import numpy, networkx
    from assemblyfire.assemblies import AssemblyGroup

    all_groups = list(assembly_grp_dict.keys())
    to_idx = {}; from_idx = []
    i = 0
    for grp_label in all_groups:
        for assembly in assembly_grp_dict[grp_label]:
            to_idx[(grp_label, assembly.idx)] = i
            from_idx.append((grp_label, assembly.idx))
            i += 1
    L = numpy.max(list(to_idx.values())) + 1
    M = numpy.zeros((L, L))
    all_gids = numpy.unique(numpy.hstack([assembly_grp.all for assembly_grp in assembly_grp_dict.values()]))

    for ref_label in all_groups:  # Step 1.
        print("Using {0} as reference...".format(ref_label))
        aligned, aligned_scores, _ = evaluate_alignment_over_seeds(assembly_grp_dict,
                                                                   ref_label,
                                                                   score_function=score_function,
                                                                   return_aligned=True)
        for aligned_label, aligned_grp in aligned.items():  # Step 2.
            for row_assembly, col_assembly, scores in zip(assembly_grp_dict[ref_label],
                                                          aligned_grp,
                                                          aligned_scores[aligned_label]):
                row_idx = to_idx[(ref_label, row_assembly.idx)]
                col_idx = to_idx[(aligned_label, col_assembly.idx)]
                M[row_idx, col_idx] += scores
    #  TODO: One thing we can do: Check overall consistency by comparing M to M.transpose()

    G = networkx.from_numpy_array(M > threshold)  # Step 3.
    components = list(map(list, networkx.connected_components(G)))

    out_lst = []
    out_lst_core = []
    consensus_origin = []
    for i, component in enumerate(components):  # Step 4.
        grp_label, assembly_index = from_idx[component[0]]
        new_assembly = assembly_grp_dict[grp_label].assemblies[assembly_index]
        new_assembly_core = new_assembly
        consensus_origin.append([(grp_label, assembly_index)])
        for idx in component[1:]:
            grp_label, assembly_index = from_idx[idx]
            new_assembly = new_assembly + assembly_grp_dict[grp_label].assemblies[assembly_index]
            new_assembly_core = new_assembly_core * assembly_grp_dict[grp_label].assemblies[assembly_index]
            consensus_origin[-1].append((grp_label, assembly_index))
        new_assembly.idx = i
        new_assembly_core.idx = i
        out_lst.append(new_assembly)
        out_lst_core.append(new_assembly_core)

    return (AssemblyGroup(out_lst, all_gids, label="Consensus assemblies"),
            AssemblyGroup(out_lst_core, all_gids, label="Core assemblies"),
            consensus_origin
            )