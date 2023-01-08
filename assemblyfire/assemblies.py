"""
Classes to handle cell assemblies detected in the previous step of the pipeline
authors: Michael Reimann, András Ecker, Daniela Egas Santander
last modified: 10.2022
"""

import os
import numpy as np
from scipy.stats import binom, pearsonr, hypergeom


__io_version__ = "1.0"
__str_io_version__ = "_version"
__h5_strings__ = {"1.0": {
    "assembly_group": "assemblies",
    "gids": "all_gids",
    "bool_index": "assembly_gids",
    "indices": "assembly_indices",
    "consensus": "consensus_assemblies",
    "instantiations_label": "instantiations",
    "consensus_magic_string": "_consensus"
    }
}
__RESERVED__ = []


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

    for i, assembly in enumerate(data):
        if isinstance(assembly, ConsensusAssembly):
            __consensus_to_h5_1p0__(assembly, grp_out, prefix=strings["consensus"], label=str(i))

    grp_out.attrs[strings["indices"]] = np.array([assembly.idx if assembly.idx is not None else -1
                                                  for assembly in data])
    return prefix


def __from_h5_1p0__(h5, group_name, prefix=None):
    strings = __h5_strings__["1.0"]
    if prefix is None:
        prefix = strings["assembly_group"]

    prefix_grp = h5[prefix]
    assert group_name in prefix_grp.keys()
    all_neurons = np.unique(np.hstack([prefix_grp[k][strings["gids"]][:]
                                       for k in prefix_grp.keys()
                                       if k not in __RESERVED__]))

    R = prefix_grp[group_name][strings["gids"]][:]
    M = prefix_grp[group_name][strings["bool_index"]][:]
    metadata = dict(prefix_grp[group_name].attrs)
    orig_indices = metadata.get(strings["indices"], list(range(M.shape[1])))

    consensus_groups = []
    if strings["consensus"] in prefix_grp[group_name]:
        consensus_groups = list(prefix_grp[group_name][strings["consensus"]].keys())

    final_assemblies = []
    for i, idx in enumerate(orig_indices):
        assembly = Assembly(R[M[:, i].astype(bool)], index=idx)
        if str(i) in consensus_groups:
            cons_assembly = __consensus_from_h5_1p0__(prefix_grp[group_name], str(i), prefix=strings["consensus"])
            assert cons_assembly.idx == assembly.idx, "Consistency check"
            assert len(cons_assembly.gids) == len(assembly.gids), "Consistency check"
            final_assemblies.append(cons_assembly)
        else:
            final_assemblies.append(assembly)

    return AssemblyGroup(final_assemblies, all_neurons, label=group_name, metadata=metadata)


def __consensus_to_h5_1p0__(data, h5, prefix=None, label=None):
    strings = __h5_strings__["1.0"]
    if prefix is None:
        prefix = strings["consensus"]
    if label is None:
        prefix = prefix + "/" + str(data.label)
    else:
        prefix = prefix + "/" + label
    consensus_metadata = {
        "label": str(data.label),
        "index": data.idx,
        "core_threshold": data._thresh,
        "core_method": data._core_method,
        strings["consensus_magic_string"]: True
    }
    instantiations = AssemblyGroup(data.instantiations, data.union.gids,
                                   label=strings["instantiations_label"])
    __to_h5_1p0__(instantiations, h5, prefix=prefix)
    __meta_to_h5_1p0__(consensus_metadata, h5, prefix=prefix)


def __consensus_from_h5_1p0__(h5, group_name, prefix=None):
    strings = __h5_strings__["1.0"]
    if prefix is None:
        prefix = strings["consensus"]

    prefix_grp = h5[prefix]
    assert group_name in prefix_grp.keys()
    grp_meta = __meta_from_h5_1p0__(prefix_grp, prefix=group_name)
    assert grp_meta.get(strings["consensus_magic_string"], False), \
        "Assembly {0} at {1} not a consensus assembly!".format(group_name, prefix)
    grp_meta.pop(strings["consensus_magic_string"])

    instantiations = __from_h5_1p0__(prefix_grp, strings["instantiations_label"], prefix=group_name)

    return ConsensusAssembly(list(instantiations), **grp_meta)


def __meta_to_h5_1p0__(metadata, h5, prefix=None):
    strings = __h5_strings__["1.0"]
    if prefix is None:
        prefix = strings["assembly_group"]
    grp = h5.require_group(prefix)
    for k, v in metadata.items():
        grp.attrs[k] = v
    return prefix


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
__cons_writers__ = {
    "1.0": __consensus_to_h5_1p0__
}
__cons_readers__ = {
    "1.0": __consensus_from_h5_1p0__
}


class Assembly(object):
    def __init__(self, lst_gids, index=None):
        self.gids = np.array(lst_gids)
        if hasattr(index, "__len__"):
            self.idx = tuple(index)
        elif index is None:
            self.idx = -1
        else:
            self.idx = index

    def __len__(self):
        """Returns the number of contained neurons"""
        return len(self.gids)

    def __mul__(self, other):
        """
        Intersection of two assemblies
        :param other: another Assembly object
        :return: An Assembly object containing neurons that are both in this and the other assembly
        """
        return Assembly(np.intersect1d(self.gids, other.gids))

    def __add__(self, other):
        """
        Union of assemblies
        :param other: another Assembly object
        :return: n Assembly object containing neurons that are either in this or the other assembly
        """
        return Assembly(np.unique(np.hstack([self.gids, other.gids])))

    def __iter__(self):
        return self.gids.__iter__()

    def to_dict(self):
        return {"gids": self.gids,
                "idx": self.idx}

    def random_subsample(self, subsample_at, seed=None):
        """
        Generate a randomly subsampled assembly
        :param subsample_at: If (int): Number of gids to sample; if (float) fraction of gids to sample
        :param seed: if specified, sets the random seed
        :return: An Assembly object containing a randomly subsampled set of gids
        """
        if seed is not None:
            np.random.seed(seed)
        if isinstance(subsample_at, int):
            N = subsample_at
        elif isinstance(subsample_at, float):
            N = int(len(self) * subsample_at)
        return Assembly(np.random.choice(self.gids, N, replace=False), index=self.idx)

    @staticmethod
    def random_numerical_gids(nrn, num_var, ref_gids, n_bins, seed):
        """Quick and dirty reimplementation of conntility's MatrixNodeIndexer functionality"""
        hist, bin_edges = np.histogram(nrn.loc[nrn["gid"].isin(ref_gids), num_var].to_numpy(), n_bins)
        bin_idx = np.digitize(nrn[num_var].to_numpy(), bin_edges)
        all_gids, sample_gids = nrn["gid"].to_numpy(), []
        for i in range(n_bins):
            if seed is not None:
                np.random.seed(seed)
            idx = np.where(bin_idx == i + 1)[0]
            sample_gids.extend(np.random.choice(all_gids[idx], hist[i], replace=False).tolist())
        return np.array(sample_gids)

    def random_numerical_control(self, nrn, num_var, n_bins=50, seed=None):
        """
        Generates random control assembly from a bigger group of gids passed,
        that have the same distribution of a (binned) numerical variable as the assembly gids
        :param nrn: pandas DataFrame with at least 2 columns: 'gid' and `num_var`
        :param num_var: numerical variable to use for sampling (has to be present in `nrn`)
        :param n_bins: number of bins to use to bin assembly `num_var`
        :param seed: if specified, sets the random seed
        :return: An Assembly object containing a randomly sampled set of gids
        """
        assert np.in1d(self.gids, nrn["gid"].to_numpy(), assume_unique=True).all(), "Not all assembly gids are part" \
                                                                                    "of the DataFrame passed"
        return Assembly(self.random_numerical_gids(nrn, num_var, self.gids, n_bins, seed), index=self.idx)

    @staticmethod
    def random_categorical_gids(nrn, cat_var, ref_gids, seed):
        """Quick and dirty reimplementation of conntility's MatrixNodeIndexer functionality"""
        values, counts = np.unique(nrn.loc[nrn["gid"].isin(ref_gids), cat_var].to_numpy(), return_counts=True)
        all_gids, cat_vals, sample_gids = nrn["gid"].to_numpy(), nrn[cat_var].to_numpy(), []
        for value, count in zip(values, counts):
            if seed is not None:
                np.random.seed(seed)
            sample_gids.extend(np.random.choice(all_gids[cat_vals == value], count, replace=False).tolist())
        return np.array(sample_gids)

    def random_categorical_control(self, nrn, cat_var, seed=None):
        """
        Generates random control assembly from a bigger group of gids passed,
        that have the same distribution of a categorical variable as the assembly gids
        :param nrn: pandas DataFrame with at least 2 columns: 'gid' and `cat_var`
        :param cat_var: categorical variable to use for sampling (has to be present in `nrn`)
        :param seed: if specified, sets the random seed
        :return: An Assembly object containing a randomly sampled set of gids
        """
        assert np.in1d(self.gids, nrn["gid"].to_numpy(), assume_unique=True).all(), "Not all assembly gids are part" \
                                                                                    "of the DataFrame passed"
        return Assembly(self.random_categorical_gids(nrn, cat_var, self.gids, seed), index=self.idx)


class AssemblyGroup(object):
    h5_read_func = __h5_readers__
    h5_write_func = __h5_writers__

    def __init__(self, assemblies, all_gids, label=None, metadata=None):
        all_gids = np.array(all_gids)
        assert label not in __RESERVED__, "{0} is a reserved name".format(label)
        if isinstance(assemblies, list):
            self.assemblies = assemblies
        elif isinstance(assemblies, np.ndarray) and assemblies.ndim == 2:
            self.assemblies = [Assembly(all_gids[assemblies[:, i]], index=i) for i in range(assemblies.shape[1])]
        else:
            raise ValueError("Specify either a list of Assembly objects or a boolean matrix of assembly membership!")

        idx_type = [isinstance(asmbl.idx, tuple) for asmbl in self.assemblies]
        assert np.mod(np.sum(idx_type), len(idx_type)) == 0, "Assembly.idx must be all tuples or all scalar in a group!"
        self.label = label
        self.all = all_gids
        if metadata is None:
            self.metadata = {}
        else:
            self.metadata = metadata

    def __iter__(self):
        """Returns an iterator over contained Assemblies"""
        return self.assemblies.__iter__()

    def __len__(self):
        """Returns the number of contained assemblies"""
        return len(self.assemblies)

    def __add__(self, other):
        """
        Concatenation of AssemblyGroups
        :param other: another AssemblyGroup object
        :return: An AssemblyGroup that is the concatenation of the Assemblies in this and the other
        """
        new_meta = {"parent 1": {"label": self.label, "metadata": self.metadata.copy()},
                    "parent 2": {"label": other.label, "metadata": other.metadata.copy()},
                    "operation": "+"}
        return AssemblyGroup(self.assemblies + other.assemblies,
                             np.union1d(self.all, other.all),
                             label=str(self.label) + " + " + str(other.label),
                             metadata=new_meta)

    def __mul__(self, other):
        """
        Aligned intersection with other
        :param other: another AssemblyGroup
        :return: the aligned intersections of this with other
        """
        return self.aligned_intersections(other)

    def to_dict(self):
        return {"label": str(self.label),
                "all_gids": list(self.all),
                "assemblies": [_assembly.to_dict() for _assembly in self],
                "metadata": self.metadata}

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
            return np.in1d(self.all, assembly.gids)
        return np.vstack([self.as_bool(iloc=i) for i in range(len(self))]).transpose()

    def loc(self, idx):
        matches = [assembly for assembly in self if assembly.idx == idx]
        if len(matches) == 0:
            raise ValueError("Assembly #{0} not found!".format(idx))
        return matches[0]

    def iloc(self, idx):
        return self.assemblies[idx]

    def union(self):
        """
        :return: An Assembly object representing the union of all contained assemblies
        """
        assert len(self) > 0
        ret = self.iloc(0)
        for i in range(1, len(self)):
            ret = ret + self.iloc(i)
        return ret

    def lengths(self):
        """Returns sizes of contained Assembly objects"""
        return np.array(list(map(len, self)))

    def random_control_from_union(self):
        """Returns an AssemblyGroup object that represents a random control. First we get the union of all Assemblies
        contained in this group. Then we randomly subsample the union, at the sizes of all Assemblies contained.
        The result is an AssemblyGroup with the same number and sizes of Assemblies as this, but Assembly membership
        is shuffled."""
        union = self.union()
        random_lst = [union.random_subsample(len(assembly)) for assembly in self]
        new_meta = {"parent": {"label": self.label, "metadata": self.metadata.copy()},
                    "operation": "shuffled"}
        return AssemblyGroup(random_lst, self.all, label=self.label + " -- shuffled", metadata=new_meta)

    def random_numerical_controls(self, nrn, num_var, n_bins=50, seed=None):
        """Returns an AssemblyGroup object that represents a random control
        with the same number and sizes of Assemblies as this, but Assembly membership is random
        and depends on a (binned) numerical variable (e.g. depth).
        See `Assembly.random_numerical_control()` above"""
        ctrl_lst = [assembly.random_numerical_control(nrn, num_var, n_bins, seed) for assembly in self]
        all_gids = np.unique(np.concatenate([assembly.gids for assembly in self]))
        new_meta = {"parent": {"label": self.label, "metadata": self.metadata.copy()},
                    "operation": "numerical random control"}
        return AssemblyGroup(ctrl_lst, all_gids, label=self.label + " -- num. control", metadata=new_meta)

    def random_categorical_controls(self, nrn, cat_var, seed=None):
        """Returns an AssemblyGroup object that represents a random control
        with the same number and sizes of Assemblies as this, but Assembly membership is random
        and depends on a categorical variable (e.g. layer, mtype etc.).
        See `Assembly.random_categorical_control()` above"""
        ctrl_lst = [assembly.random_categorical_control(nrn, cat_var, seed) for assembly in self]
        all_gids = np.unique(np.concatenate([assembly.gids for assembly in self]))
        new_meta = {"parent": {"label": self.label, "metadata": self.metadata.copy()},
                    "operation": "categorical random control"}
        return AssemblyGroup(ctrl_lst, all_gids, label=self.label + " -- cat. control", metadata=new_meta)

    def to_h5(self, filename, prefix=None, version=None):
        """
        :param filename: Filename to write this assembly group to
        :param prefix: Default: None, a prefix within the file to put the data behind
        :param version: default: latest
        :return: str: the prefix used
        """
        import h5py
        write_func = self.__class__.h5_write_func[__initialize_h5__(filename, version=version)]
        with h5py.File(filename, "r+") as h5:
            return write_func(self, h5, prefix)

    @classmethod
    def from_h5(cls, fn, group_name, prefix=None):
        import h5py
        read_func = cls.h5_read_func[__initialize_h5__(fn, assert_exists=True)]
        with h5py.File(fn, "r") as h5:
            return read_func(h5, group_name, prefix=prefix)

    def intersection_sizes(self, other=None):
        """
        Pairwise intersections
        :param other: Another AssemblyGroup object. If none provided, this object is used
        :return: (numpy.array) For all combinations of an assembly from this and the other group the
        size of the intersecting Assembly
        """
        if other is None:
            return self.intersection_sizes(self)

        return np.array([[len(a * b) for b in other]
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
        new_all = np.union1d(self.all, other.all)
        new_meta = {"parent 1": {"label": self.label, "metadata": self.metadata.copy()},
                    "parent 2": {"label": other.label, "metadata": other.metadata.copy()},
                    "operation": "*"}
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
        expected_sd = np.sqrt(self.expected_intersection_sizes(other, moment='v'))
        return (actual - expected_mn) / (expected_sd + epsilon)

    def expected_intersection_sizes(self, other=None, moment="m"):
        """
        :param other: Another AssemblyGroup object. If none provided, this object is used
        :param moment: Which statistical moment (see scipy.stats; "m" = mean, "v" = variance)
        :return: (numpy.array) The matrix of expected values for the size of overlaps between assemblies, based on
        their respective sizes and the ... Here, for all pairs of assemblies in this and other
        """
        if other is None:
            return self.expected_intersection_sizes(self)

        return np.array([[hypergeom(len(self.all), len(a), len(b)).stats(moment) for b in other]
                          for a in self])

    @staticmethod
    def norm_correlation(A, B):
        """
        Normalized correlations between values in A and B; rows: samples/observations; columns: variables
        :param A: A numpy.array of shape N x A
        :param B: A numpy.array of shape N x B, i.e. the same first dimension as A
        :return: numpy.array of shape A x B
        """
        A = A - A.mean(axis=0, keepdims=True)
        B = B - B.mean(axis=0, keepdims=True)
        M = np.dot(A.transpose(), B) / A.shape[0]
        M = M / np.sqrt(np.dot(A.var(axis=0, keepdims=True).transpose(),  # N x 1
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
        return self.norm_correlation(I, O)


class WeightedAssembly(Assembly):
    """Represents an assembly with weights on neurons. Weights can be given in different ways, by activity,
    by network properties, or by coreness in the ConsensusAssembly child class"""

    # TODO: Maybe we want to add read/write functions if the weights come from network metrics that are hard to compute.
    def __init__(self, lst_gids, weights):
        """
        :param weights: A list of the same size of the assembly. It represents the weights of neurons used to compute a
        filtration.
        """
        assert len(lst_gids) == len(weights), "The list of gids and the list of weights must be of the same length"
        self.gids = np.array(lst_gids)
        self.weights = np.array(weights)

    def at_weight(self, threshold, method="strength"):
        """ Returns thresholded assembly
        :param method: distance returns gids with weight smaller or equal than thresh
                       strength returns gids with weight larger or equal than tresh"""
        if method == "strength":
           return Assembly(self.gids[self.weights >= threshold])
        elif method == "distance":
           return Assembly(self.gids[self.weights <= threshold])
        else:
            raise ValueError("Method has to be 'strength' or 'distance'")

    def filtration(self, method='strength'):
        """Returns an AssemblyGroup object represeting the filtration of that assembly.
        :param method:distance smaller weights enter the filtration first
                      strength larger weights enter the filtration first"""
        if method == "strength":
            filtration_weights = np.unique(self.weights)[::-1]
        elif method == "distance":
            filtration_weights = np.unique(self.weights)
        else:
           raise ValueError("Method has to be 'strength' or 'distance'")
        filtration = [self.at_weight(filtration_weight, method=method) for filtration_weight in filtration_weights]
        return AssemblyGroup(filtration, self.gids, label=None, metadata=None)


class ConsensusAssembly(Assembly):
    """
    Represents an assembly that is a "consensus" of several stochastic runs. It is instantiated by providing
    the list of assemblies constituting it in the individual stochastic runs. It calculates for each neuron
    the number of stochastic runs it shows up in, and based on that calculates a "coreness" property. We consider
    a neuron to be part of this assembly, if its "coreness" property is above a user-specified threshold
    """
    h5_read_func = __cons_readers__
    h5_write_func = __cons_writers__

    def __init__(self, lst_assemblies, index=None, label=None, core_threshold=4.0, core_method="p-value"):
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
        self.label = label
        if core_method == "raw":
            self.coreness = self.__number_of_times_contained__()
        elif core_method == "number":
            self.coreness = self.calculate_coreness(self.__number_of_times_contained__(),
                                                    expected_n=self.__expected_number_of_instantiations__())
        elif core_method == "p-value":
            self.coreness = self.calculate_coreness(self.__number_of_times_contained__(),
                                                    expected_distribution=self.__expected_distribution_of_instantiations__())
        else:
            raise ValueError("Need to specify one of 'raw', 'number' or 'p-value' for core_method")
        super(ConsensusAssembly, self).__init__(self.union.gids[self.coreness > self._thresh], index=index)

    def at_threshold(self, new_thresh, core_method=None):
        """
        :param new_thresh: new threshold in terms of "coreness" that determines consensus cluster membership
        :param core_method: (Optional) instantiate the copy using a different core_method
        :return:
        """
        if core_method is None:
            core_method = self._core_method
        return ConsensusAssembly(self.instantiations, index=self.idx,
                                 core_method=core_method, core_threshold=new_thresh)
    
    def at_size_preserving_threshold(self):
        return self.at_threshold(self.__size_preserving_threshold__())

    @staticmethod
    def calculate_coreness(vec_num_contained, expected_n=None, expected_distribution=None, epsilon=1E-5):
        if expected_n is not None:
            return (vec_num_contained - expected_n) / (vec_num_contained + expected_n)
        if expected_distribution is not None:
            p_values = 1.0 + epsilon - expected_distribution.cdf(vec_num_contained)
            return -np.log10(p_values)
        raise ValueError("Need to specify 'expected_n' or 'expected_distribution'!")

    def to_h5(self, filename, prefix=None, version=None):
        """
        :param filename: Filename to write this assembly group to
        :param prefix: Default: None, a prefix within the file to put the data behind
        :param version: default: latest
        :return: str: the prefix used
        """
        import h5py
        write_func = self.__class__.h5_write_func[__initialize_h5__(filename, version=version)]
        with h5py.File(filename, "r+") as h5:
            return write_func(self, h5, prefix)

    @classmethod
    def from_h5(cls, fn, group_name, prefix=None):
        import h5py
        read_func = cls.h5_read_func[__initialize_h5__(fn, assert_exists=True)]
        with h5py.File(fn, "r") as h5:
            return read_func(h5, group_name, prefix=prefix)
    
    def __size_preserving_threshold__(self):
        L = np.mean(list(map(len, self.instantiations)))
        P = 100 - 100 * L / len(self.union)
        return np.percentile(self.coreness, P)

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
        p_idv = np.array([len(_x) for _x in self.instantiations]) / len(self.union)
        return np.sum(p_idv)

    def __expected_distribution_of_instantiations__(self):
        """
        :return: Based on the size of the union Assembly and the mean size of individual instantiations, what is the
        expected distribution of the number of times a given neuron would be part of an instantiations
        in a random control?
        """
        N = len(self.instantiations)
        p = np.array([len(_x) for _x in self.instantiations]).mean() / len(self.union)
        return binom(N, p)

    def __number_of_times_contained__(self):
        res = []
        for assembly in self.instantiations:
            res.append(np.in1d(self.union.gids, assembly.gids))
        return np.vstack(res).sum(axis=0)


def build_assembly_group(gids, n_assemblies, assembly_lst, seeds, assembly_grp_dict):
    """
    Builds 1 big assembly group from assemblies in `assembly_grp_dict` for consensus clustering
    (`gids`, `n_assemblies`, `assembly_lst`, and `seeds` are passed to be able to create consensus
    from a subset of seeds only and then increase the size of the subset)
    """
    for seed in seeds:
        assembly_grp = assembly_grp_dict[seed]
        gids.extend(assembly_grp.all.tolist())
        n = len(assembly_grp.assemblies)
        n_assemblies.append(n)
        assembly_lst.extend([assembly_grp.assemblies[i] for i in range(n)])
    all_gids = np.unique(gids)
    assembly_grp = AssemblyGroup(assemblies=assembly_lst, all_gids=all_gids, label="all")
    return gids, n_assemblies, assembly_lst, assembly_grp


def consensus_over_seeds(assembly_grp_dict, h5f_name, h5_prefix, fig_path,
                         distance_metric="jaccard", linkage_method="ward"):
    """
    Hierarhichal clustering of assemblies from different seeds based on specified distance metric
    :param assembly_grp_dict: dict with seeds as keys and AssemblyGroup object as values
    :param distance_metric, linkage_method: see `clustering.py/cluster_assemblies()`
    :return: assembly_grp_clust: dict with cluster idx as keys and AssemblyGroup object as values
    """
    from assemblyfire.clustering import cluster_assemblies
    from assemblyfire.plots import plot_assembly_sim_matrix, plot_dendogram_silhouettes

    gids, n_assemblies, assembly_lst = [], [], []
    seeds = list(assembly_grp_dict.keys())
    _, n_assemblies, _, assembly_grp = build_assembly_group(gids, n_assemblies, assembly_lst, seeds, assembly_grp_dict)

    sim_matrix, clusters, plotting = cluster_assemblies(assembly_grp.as_bool().T, n_assemblies,
                                                        distance_metric, linkage_method)
    # plotting clustering results
    fig_name = os.path.join(fig_path, "simmat_assemblies_%s.png" % distance_metric)
    plot_assembly_sim_matrix(sim_matrix.copy(), n_assemblies, fig_name)
    fig_name = os.path.join(fig_path, "%s_clustering_assemblies.png" % linkage_method)
    plot_dendogram_silhouettes(clusters, *plotting, fig_name)

    # making consensus assemblies from assemblies grouped by clustering
    for cluster in np.unique(clusters):
        c_idx = np.where(clusters == cluster)[0]
        assembly_lst = [assembly_grp.assemblies[i] for i in c_idx]
        cons_assembly = ConsensusAssembly(assembly_lst, index=cluster, label="cluster%i" % cluster)
        cons_assembly.to_h5(h5f_name, prefix=h5_prefix)

