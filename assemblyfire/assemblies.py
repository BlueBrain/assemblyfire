# -*- coding: utf-8 -*-
"""
Classes to handle cell assemblies detected in the previous step of the pipeline
authors: Michael Reimann, András Ecker, Daniela Egas Santander
last modified: 11.2020
"""

import numpy
from scipy.stats import binom, pearsonr, hypergeom


__io_version__ = "1.0"
__str_io_version__ = "_version"
__h5_strings__ = {"1.0": {
    "assembly_group": "assemblies",
    "gids": "all_gids",
    "bool_index": "assembly_gids",
    "indices": "assembly_indices"
    }
}
__RESERVED__ = []


def __to_h5_1p0__(data, h5, prefix=None):
    strings = __h5_strings__["1.0"]
    if prefix is None:
        prefix = strings["assembly_group"]
    grp = h5.require_group(prefix)
    assert data.label not in grp, "{0} already in {1}/{2}".format(data.label, h5.filename, prefix)
    grp_out = grp.create_group(data.label)

    grp_out.create_dataset(strings["gids"], data=data.all)
    grp_out.create_dataset(strings["bool_index"], data=data.as_bool())
    for k, v in data.metadata.items():
        grp_out.attrs[k] = v
    grp_out.attrs[strings["indices"]] = [assembly.idx if assembly.idx is not None else -1
                                         for assembly in data]
    return prefix


def __from_h5_1p0__(h5, group_name, prefix=None):
    strings = __h5_strings__["1.0"]
    if prefix is None:
        prefix = strings["assembly_group"]

    prefix_grp = h5[prefix]
    assert group_name in prefix_grp.keys()
    all_neurons = numpy.unique(numpy.hstack([prefix_grp[k][strings["gids"]][:]
                                             for k in prefix_grp.keys()]))

    R = prefix_grp[group_name][strings["gids"]][:]
    M = prefix_grp[group_name][strings["bool_index"]][:]
    metadata = dict(prefix_grp[group_name].attrs)
    orig_indices = metadata.get(strings["indices"], list(range(M.shape[1])))
    assemblies = [Assembly(R[M[:, i].astype(bool)], index=idx)
                  for i, idx in enumerate(orig_indices)]
    return AssemblyGroup(assemblies, all_neurons, label=group_name, metadata=metadata)


def __meta_from_h5_1p0__(h5, group_name=None, prefix=None):
    strings = __h5_strings__["1.0"]
    if prefix is None:
        prefix = strings["assembly_group"]

    prefix_grp = h5[prefix]
    metadata = dict(prefix_grp.attrs)
    if group_name is not None:
        assert group_name in prefix_grp.keys()
        metadata.update(dict(prefix_grp[group_name].attrs))
    return metadata


def __meta_to_h5_1p0__(metadata, h5, prefix=None):
    strings = __h5_strings__["1.0"]
    if prefix is None:
        prefix = strings["assembly_group"]
    grp = h5.require_group(prefix)
    for k, v in metadata.items():
        grp.attrs[k] = v
    return prefix


__h5_writers__ = {
    "1.0": __to_h5_1p0__
}
__h5_readers__ = {
    "1.0": __from_h5_1p0__
}
__meta_readers__ = {
    "1.0": __meta_from_h5_1p0__
}
__meta_writers__ = {
    "1.0": __meta_to_h5_1p0__
}


class AssemblyProjectMetadata(object):
    #TODO decide how to structure metadata and write a proper Class
    @staticmethod
    def from_h5(fn, group_name=None, prefix=None):
        import h5py
        read_func = __meta_readers__[AssemblyGroup.__initialize_h5__(fn, assert_exists=True)]
        with h5py.File(fn, "r") as h5:
            meta_dict = read_func(h5, group_name=group_name, prefix=prefix)
        return meta_dict

    @staticmethod
    def to_h5(metadata, fn, prefix=None, version=None):
        import h5py
        write_func = __meta_writers__[AssemblyGroup.__initialize_h5__(fn, version=version)]
        with h5py.File(fn, "r+") as h5:
            return write_func(metadata, h5, prefix=prefix)


class Assembly(object):
    def __init__(self, lst_gids, index=None):
        self.gids = numpy.array(lst_gids)
        self.idx = index

    def __len__(self):
        """
        :return: (int) Number of contained neurons
        """
        return len(self.gids)

    def __mul__(self, other):
        """
        Intersection of two assemblies
        :param other: another Assembly object
        :return: An Assembly object containing neurons that are both in this and the other assembly
        """
        return Assembly(numpy.intersect1d(self.gids, other.gids))

    def __add__(self, other):
        """
        Union of assemblies
        :param other: another Assembly object
        :return: n Assembly object containing neurons that are either in this or the other assembly
        """
        return Assembly(numpy.unique(numpy.hstack([self.gids, other.gids])))

    def __iter__(self):
        return self.gids.__iter__()

    def to_dict(self):
        return {
            "gids": self.gids,
            "idx": self.idx
        }

    def random_subsample(self, subsample_at, seed=None):
        """
        Generate a randomly subsampled assembly
        :param subsample_at: If (int): Number of gids to sample; if (float) fraction of gids to sample
        :param seed: if specified, sets the random seed
        :return: An Assembly object containing a randomly subsampled set of gids
        """
        if seed is not None:
            numpy.random.seed(seed)
        if isinstance(subsample_at, int):
            N = subsample_at
        elif isinstance(subsample_at, float):
            N = int(len(self) * subsample_at)
        return Assembly(numpy.random.choice(self.gids, N, replace=False), index=self.idx)


class AssemblyGroup(object):
    def __init__(self, assemblies, all_gids, label=None, metadata=None):
        all_gids = numpy.array(all_gids)
        assert label not in __RESERVED__, "{0} is a reserved name".format(label)
        if isinstance(assemblies, list):
            self.assemblies = assemblies
        elif isinstance(assemblies, numpy.ndarray) and assemblies.ndim == 2:
            self.assemblies = [Assembly(all_gids[assemblies[:, i]], index=i) for i in range(assemblies.shape[1])]
        else:
            raise ValueError("Specify either a list of Assembly objects or a boolean matrix of assembly membership!")
        self.label = label
        self.all = all_gids
        if metadata is None:
            self.metadata = {}
        else:
            self.metadata = metadata

    def union(self):
        """
        :return: An Assembly object representing the union of all contained assemblies
        """
        assert len(self) > 0
        ret = self.iloc(0)
        for i in range(1, len(self)):
            ret = ret + self.iloc(i)
        return ret

    def random_control_from_union(self):
        """
        :return: An AssemblyGroup object that represents a random control. First we get the union of all Assemblies
        contained in this group. Then we randomly subsample the union, at the sizes of all Assemblies contained.
        The result is an AssemblyGroup with the same number and sizes of Assemblies as this, but Assembly mambership
        is shuffled.
        """
        union = self.union()
        lst_random = [union.random_subsample(len(assembly)) for assembly in self]
        new_metadata = {
            "parent": {
                "label": self.label,
                "metadata": self.metadata.copy()
            },
            "operation": "shuffled"
        }
        return AssemblyGroup(lst_random, self.all, label=self.label + " -- shuffled", metadata=new_metadata)

    def __iter__(self):
        """
        :return: iterator over contained Assemblies
        """
        return self.assemblies.__iter__()

    def __len__(self):
        """
        :return: number of contained assemblies
        """
        return len(self.assemblies)

    def __add__(self, other):
        """
        Concatenation of AssemblyGroups
        :param other: another AssemblyGroup object
        :return: An AssemblyGroup that is the concatenation of the Assemblies in this and the other
        """
        new_meta = {
            "parent 1": {
                "label": self.label,
                "metadata": self.metadata.copy()
            },
            "parent 2": {
                "label": other.label,
                "metadata": other.metadata.copy()
            },
            "operation": "+"
        }
        return AssemblyGroup(self.assemblies + other.assemblies,
                             numpy.union1d(self.all, other.all),
                             label=str(self.label) + " + " + str(other.label),
                             metadata=new_meta)

    def __mul__(self, other):
        """
        Aligned intersection with other
        :param other: another AssemblyGroup
        :return: the aligned intersections of this with other
        """
        return self.aligned_intersections(other)

    def as_bool(self, loc=None, iloc=None):
        """
        Returns a bool vector where an entry is true if a neuron from self.all is contained in an assembly.
        :param iloc: (optional) if provided, return 1d boolean representation of assembly at that index,
        else a 2d matrix where the columns are the boolean representations of all assemblies contained in this group.
        :param loc: (optional) if provided, return 1d boolean representation of assembly with that .idx property
        :return: numpy.array
        """
        assembly = None
        if iloc is not None:
            assembly = self.iloc(iloc)
        elif loc is not None:
            assembly = self.loc(loc)
        if assembly is not None:
            return numpy.in1d(self.all, assembly.gids)
        return numpy.vstack([self.as_bool(iloc=i) for i in range(len(self))]).transpose()

    def loc(self, idx):
        matches = [assembly for assembly in self if assembly.idx == idx]
        if len(matches) == 0:
            raise ValueError("Assembly #{0} not found!".format(idx))
        return matches[0]

    def iloc(self, idx):
        return self.assemblies[idx]

    def to_dict(self):
        return {
            "label": str(self.label),
            "all_gids": list(self.all),
            "assemblies": [_assembly.to_dict() for _assembly in self],
            "metadata": self.metadata
        }

    @staticmethod
    def __initialize_h5__(filename, version=None, assert_exists=False):
        import h5py
        import os
        if version is None:
            version = __io_version__
        existed = os.path.exists(filename)
        assert existed or not assert_exists, "File {0} does not exist!".format(filename)
        if not existed:
            assert version in __h5_writers__, "Unknown version: {0}".format(version)
        with h5py.File(filename, "a") as h5:
            if existed:
                # assert __str_io_version__ in h5.attrs, "Invalid file" TODO: Put that back in
                version = h5.attrs.get(__str_io_version__, __io_version__)
                assert version in __h5_writers__, "Unknown version: {0}".format(version)
            else:
                h5.attrs[__str_io_version__] = version

        return version

    def to_h5(self, filename, prefix=None, version=None):
        """
        :param filename: Filename to write this assembly group to
        :param prefix: Default: None, a prefix within the file to put the data behind
        :param version: default: latest
        :return: str: the prefix used
        """
        import h5py
        write_func = __h5_writers__[self.__initialize_h5__(filename, version=version)]
        with h5py.File(filename, "r+") as h5:
            return write_func(self, h5, prefix)

    @staticmethod
    def from_h5(fn, group_name, prefix=None):
        import h5py
        read_func = __h5_readers__[AssemblyGroup.__initialize_h5__(fn, assert_exists=True)]
        with h5py.File(fn, "r") as h5:
            return read_func(h5, group_name, prefix=prefix)

    def lengths(self):
        """
        :return: (numpy.array) sizes of contained Assembly objects
        """
        return numpy.array(list(map(len, self)))

    def intersection_sizes(self, other=None):
        """
        Pairwise intersections
        :param other: Another AssemblyGroup object. If none provided, this object is used
        :return: (numpy.array) For all combinations of an assembly from this and the other group the
        size of the intersecting Assembly
        """
        if other is None:
            return self.intersection_sizes(self)

        return numpy.array([[len(a * b) for b in other]
                            for a in self])

    def aligned_intersections(self, other=None):
        """
        Intersections along the diagonal
        :param other: Another AssemblyGroup object. If none provided, this object is used
        :return: (AssemblyGroup) For every assembly in this group the intersection with the Assembly at the
        corresponding location of the other group
        """
        if other is None:
            return self.aligned_intersections(self)
        new_all = numpy.union1d(self.all, other.all)
        new_meta = {
            "parent 1": {
                "label": self.label,
                "metadata": self.metadata.copy()
            },
            "parent 2": {
                "label": other.label,
                "metadata": other.metadata.copy()
            },
            "operation": "*"
        }

        return AssemblyGroup([a * b for a, b in zip(self, other)],
                             new_all, label=str(self.label) + " * " + str(other.label),
                             metadata=new_meta)

    def rel_intersection_sizes(self, other=None):
        """
        :param other: Another AssemblyGroup object. If none provided, this object is used
        :return: (numpy.array) the sizes of pairwise intersections, relative to a random control
        """
        epsilon = 0.1
        actual = self.intersection_sizes(other)
        expected_mn = self.expected_intersection_sizes(other)
        expected_sd = numpy.sqrt(self.expected_intersection_sizes(other, moment='v'))
        return (actual - expected_mn) / (expected_sd + epsilon)

    def expected_intersection_sizes(self, other=None, moment="m"):
        """
        :param other: Another AssemblyGroup object. If none provided, this object is used
        :param moment: Which statistical moment (see scipy.stats; "m" = mean, "v" = variance)
        :return: (numpy.array) The matrix of expected values for the size of overlaps between assemblies, based on
        their respective sizes and the. Here, for all pairs of assemblies in this and other
        """
        if other is None:
            return self.expected_intersection_sizes(self)

        return numpy.array([[hypergeom(len(self.all), len(a), len(b)).stats(moment) for b in other]
                            for a in self])

    def expected_aligned_intersection_sizes(self, other=None, moment="m"):
        """
        :param other: Another AssemblyGroup object. If none provided, this object is used
        :param moment: Which statistical moment (see scipy.stats; "m" = mean, "v" = variance)
        :return: (numpy.array) The matrix of expected values for the size of overlaps between assemblies, based on
        their respective sizes and the. Here, for all assemblies in this group and the corresponding one in the other
        """
        if other is None:
            return self.expected_aligned_intersection_sizes(self)

        return numpy.array([hypergeom(len(self.all), len(a), len(b)).stats(moment)
                            for a, b in zip(self, other)])

    @staticmethod
    def correlation_func(A, B):
        """
        Normalized correlations between values in A and B
        rows: samples / observations
        columns: variables
        :param A: A numpy.array of shape N x A
        :param B: A numpy.array of shape N x B, i.e. the same first dimension as A
        :return: numpy.array of shape A x B
        """
        A = A - A.mean(axis=0, keepdims=True)
        B = B - B.mean(axis=0, keepdims=True)
        M = numpy.dot(A.transpose(), B) / A.shape[0]
        M = M / numpy.sqrt(numpy.dot(A.var(axis=0, keepdims=True).transpose(),  # N x 1
                                     B.var(axis=0, keepdims=True)))  # M x 1
        return M

    def intersection_pattern_correlation(self, other=None, normalized=True):
        """
        :param other: Another AssemblyGroup object. If none provided, this object is used
        :param normalized: default: True. If true, use pattern of normalized overlaps, else pattern of raw overlap sizes
        :return: (numpy.array) The matrix of how consistent each pair of one assembly from this group and another
        assembly from the other are. Consistency is calculated in terms of how similar the two are in their overlap
        with all assemblies in this group.
        """
        if other is None:
            return self.intersection_pattern_correlation(self)
        if normalized:
            I = self.rel_intersection_sizes()
            O = self.rel_intersection_sizes(other=other)
        else:
            I = self.intersection_sizes()
            O = self.intersection_sizes(other=other)
        return self.correlation_func(I, O)

    @staticmethod
    def greedy_alignment(score_matrix):
        """
        :param score_matrix: A numpy.array where entry at i, j denotes the quality of an alignment
        of item i of one group with item j of another group
        :return: A kind-of optimal alignment, i.e. a permutation of the items in the second group
        that leads to a kind-of optimal alignment
        """

        idx1 = list(range(score_matrix.shape[0]))
        idx2 = list(range(score_matrix.shape[1]))
        alignment = -numpy.ones(len(idx1), dtype=int)
        while len(idx1) > 0 and len(idx2) > 0:
            active_submat = score_matrix[numpy.ix_(idx1, idx2)]
            i, j = numpy.nonzero(active_submat == active_submat.max())
            alignment[idx1[i[0]]] = idx2[j[0]]
            idx1.remove(idx1[i[0]])
            idx2.remove(idx2[j[0]])
            # TODO: Deal with: idx2 is empty but idx1 not yet
        return alignment

    def align_with(self, other, return_scores=False):
        """
        :param other: AssemblyGroup
        :param return_scores: Default: False. If true, also returns a measure of the quality of alignment
        :return: An AssemblyGroup with the same Assemblies as the input, but permutated for optimal aligment with
        the assemblies in this group
        """
        new_all = numpy.union1d(self.all, other.all)
        M = self.intersection_pattern_correlation(other)
        alignment = self.greedy_alignment(M)
        M = M[:, alignment]
        out_grp = AssemblyGroup([other.assemblies[i] for i in alignment],
                                new_all, label=other.label, metadata=other.metadata)
        if return_scores:
            return out_grp, M
        return out_grp

    def evaluate_overall_alignment(self, other):
        """
        :param other: AssemblyGroup
        :return: Evaluate how well the Assemblies in this and the other group are aligned. I.e. how similar the two
        groups are. Assumes that the other AssemblyGroup is already aligned with this
        """
        I = self.rel_intersection_sizes()
        O = self.rel_intersection_sizes(other=other)
        return pearsonr(I.flatten(), O.flatten())

    def evaluate_individual_alignments(self, other, score_function="overlap"):
        """
        :param other: AssemblyGroup
        :param score_function: (str) one of "overlap" or "correlation"
        :return: Evaluate how well _each individual_ Assembly in this group is aligned with its corresponding
        Assembly in the other group. Assumes that the other AssemblyGroup is already aligned with this
        """
        if score_function == "overlap":
            overlap = self.aligned_intersections(other).lengths()
            expected_overlap_mn = self.expected_aligned_intersection_sizes(other)
            expected_overlap_sd = numpy.sqrt(self.expected_aligned_intersection_sizes(other, moment="v"))
            return (overlap - expected_overlap_mn) / expected_overlap_sd
        elif score_function == "correlation":
            M = self.intersection_pattern_correlation(other)
            return numpy.diag(M)
        else:
            raise Exception("Unknown score function: {0}".format(score_function))


#TODO: move this to a separate file (together with parts of legacy.py)
def consensus_over_seeds_hamming(assembly_grp_dict, criterion="maxclust", threshold=None):
    """
    Hierarhichal clustering (Ward linkage) of assemblies from different seeds based on Hamming distance
    :param assembly_grp_dict: dict with seeds as keys and AssemblyGroup object as values
    :param criterion: criterion for hierarchical clustering (see `clustering.py/cluster_assemblies()`)
    :param threshold: threshold to cut dendogram if criterion is "distance" (see `clustering.py/cluster_assemblies()`)
    :return: assembly_grp_clust: dict with cluster idx as keys and AssemblyGroup object as values
    """

    from assemblyfire.clustering import cluster_assemblies
    assert criterion in ["distance", "maxclust"]

    # concatenate assemblies over seed into 1 big AssemblyGroup
    gids = []
    n_assemblies = []
    assembly_lst = []
    for _, assembly_grp in assembly_grp_dict.items():
        gids.extend(assembly_grp.all.tolist())
        n = len(assembly_grp.assemblies)
        n_assemblies.append(n)
        assembly_lst.extend([assembly_grp.assemblies[i] for i in range(n)])
    all_gids = np.unique(gids)
    all_assemblies = AssemblyGroup(assemblies=assembly_lst, all_gids=all_gids, label="all")

    # hierarhichal clustering
    if criterion == "maxclust":
        sim_matrix, clusters, plotting = cluster_assemblies(all_assemblies.as_bool().T, n_assemblies,
                                                            criterion, np.max(n_assemblies))
    elif criterion == "distance":
        sim_matrix, clusters, plotting = cluster_assemblies(all_assemblies.as_bool().T, n_assemblies,
                                                            criterion, threshold)

    # making an assembly group of assemblies grouped by clustering
    assembly_grp_clust = {}
    for cluster in np.unique(clusters):
        c_idx = np.where(clusters==cluster)[0]
        assembly_lst = [assembly_group.assemblies[i] for i in c_idx]
        assembly_grp_clust[cluster] = AssemblyGroup(assemblies=assembly_lst, all_gids=all_gids,
                                                    label="cluster%i" % cluster)

    return assembly_grp_clust


class ConsensusAssembly(Assembly):
    """
    Represents an assembly that is a "consensus" of several stochastic runs. It is instantiated by providing
    the list of assemblies constituting it in the individual stochastic runs. It calculates for each neuron
    the number of stochastic runs it shows up in, and based on that calculates a "coreness" property. We consider
    a neuron to be part of this assembly, if its "coreness" property is above a user-specified threshold
    """

    def __init__(self, lst_assemblies, index=None, core_threshold=4.0, core_method="p-value"):
        """
        :param lst_assemblies: A list of assemblies that are thought to represent the same "true" assembly.
        We call them the "instantiations" of the true assembly.
        :param index: An index to attach to this Assembly
        :param core_threshold: Threshold in terms of "coreness" that determines consensus cluster membership
        :param core_method: Method to calculate "coreness". One of:
             "number": coreness is determined as the relative difference with the expected number of times a neuron
             is expected to be member of an instantiation. Values between -1 and 1
             "p-value": coreness is determined as the negative p-value of the number of times a neuron is part of an
             instantiation in a binomial control model. Values between 0 and 5.
        """
        self.instantiations = lst_assemblies
        self.union = self.__union_of_instantiations__()
        self._core_method = core_method
        self._thresh = core_threshold
        if core_method == "number":
            self.coreness = self.calculate_coreness(self.__number_of_times_contained__(),
                                                    expected_n=self.__expected_number_of_instantiations__())
        elif core_method == "p-value":
            self.coreness = self.calculate_coreness(self.__number_of_times_contained__(),
                                                    expected_distribution=self.__expected_distribution_of_instantiations__())
        else:
            raise ValueError("Need to specify one of 'number' or 'p-value' for core_method")
        super(ConsensusAssembly, self).__init__(self.union.gids[self.coreness > self._thresh], index=index)

    def at_threshold(self, new_thresh):
        """
        :param new_thresh: new threshold in terms of "coreness" that determines consensus cluster membership
        :return:
        """
        return ConsensusAssembly(self.instantiations, index=self.idx,
                                 core_method=self._core_method, core_threshold=new_thresh)

    @staticmethod
    def calculate_coreness(vec_num_contained, expected_n=None, expected_distribution=None, epsilon=1E-5):
        if expected_n is not None:
            return (vec_num_contained - expected_n) / (vec_num_contained + expected_n)
        if expected_distribution is not None:
            p_values = 1.0 + epsilon - expected_distribution.cdf(vec_num_contained)
            return -numpy.log10(p_values)
        raise ValueError("Need to specify 'expected_n' or 'expected_distribution'!")

    def __union_of_instantiations__(self):
        """
        :return: The assembly representing the union of all stochastic instantiations of the consensus assembly
        """
        assert len(self.instantiations) > 0, "Need to specify at least one Assembly for a consensus"
        union_view = self.instantiations[0]
        for instance in self.instantiations[1:]:
            union_view = union_view + instance
        return union_view

    def __expected_number_of_instantiations__(self):
        """
        :return: Based on the size of the union Assembly and the sizes of individual instantiations, what is the
        expected number of times a given neuron would be part of an instantiations in a random control?
        """
        p_idv = numpy.array([len(_x) for _x in self.instantiations]) / len(self.union)
        return numpy.sum(p_idv)

    def __expected_distribution_of_instantiations__(self):
        """
        :return: Based on the size of the union Assembly and the mean size of individual instantiations, what is the
        expected distribution of the number of times a given neuron would be part of an instantiations
        in a random control?
        """
        N = len(self.instantiations)
        p = numpy.array([len(_x) for _x in self.instantiations]).mean() / len(self.union)
        return binom(N, p)

    def __number_of_times_contained__(self):
        res = []
        for assembly in self.instantiations:
            res.append(numpy.in1d(self.union.gids, assembly.gids))
        return numpy.vstack(res).sum(axis=0)

    def plot(self, angles=None, **kwargs):
        from matplotlib import pyplot as plt
        if angles is None:
            angles = numpy.random.rand(len(self.union)) * 2 * numpy.pi
        if self._core_method == "p-value":
            r = 10 ** -self.coreness
        elif self._core_method == "number":
            r = (1 - self.coreness) / 2.0
        else:
            raise Exception()
        plt.polar(angles, r, **kwargs)




