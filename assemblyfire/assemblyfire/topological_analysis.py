import pandas

from assemblyfire import topology_implementations
from ._topological_analysis import AnalysisImplementation
from .connectivity import ConnectivityMatrix

_str_subpop = "subpopulations"  # For hdf5 serialization
_str_params = "parameters"


class TopologicalAnalysis(ConnectivityMatrix):
    def __init__(self, *args, **kwargs):
        super(TopologicalAnalysis, self).__init__(*args, **kwargs)
        self._populations = {
            'all': (self, "Default population describing the entire population")
        }
        idx = pandas.MultiIndex.from_tuples([], names=['population', 'analysis'])
        self._results = pandas.Series([], index=idx, dtype=object)
        self._analyses = {}
        self._register_analyses(topology_implementations)

    @property
    def results(self):
        return self._results

    @property
    def populations(self):
        return self._populations

    @property
    def analyses(self):
        return list(self._analyses.keys())

    def print_analyses(self, verbose=False):
        for k, v in self._analyses.items():
            print(k)
            if verbose:
                print("\t{0}".format(v.__doc__))

    def _register_analyses(self, mdl):
        for k, v in mdl.__dict__.items():
            if isinstance(v, type):  # Alternative: use inspect.isclass
                if issubclass(v, AnalysisImplementation) and k != "AnalysisImplementation":
                    self._analyses[k] = v

    def load_analyses_from_file(self, fn):
        import importlib, os, sys
        f_path, fn = os.path.split(os.path.abspath(fn))
        remove_it = False
        if f_path not in sys.path:
            sys.path.insert(0, f_path)
            remove_it = True
        module = importlib.import_module(os.path.splitext(fn)[0])
        self._register_analyses(module)
        if remove_it:
            sys.path.remove(f_path)

    def _add_result(self, res, population, label):
        idx_tuples = self._results.index.to_numpy().tolist()
        idx_tuples.append((population, label))
        idx = pandas.MultiIndex.from_tuples(idx_tuples, names=self._results.index.names)

        old_res = self._results.values.tolist()
        old_res.append(res)
        self._results = pandas.Series(old_res, index=idx)

    def name_population(self, population, label, edge=None, description=None):
        if description is None:
            description = "A custom population"
        if edge is not None:
            population = population.default(edge)
        self._populations[label] = (population, description)

    def run(self, analysis, population, *args, **kwargs):
        #  TODO: Test if it is already int he results
        if analysis not in self._analyses:
            raise ValueError("Analysis {0} unknown!".format(analysis))
        if population not in self._populations:
            raise ValueError("Population {0} undefined!".format(population))
        ana_obj = self._analyses[analysis](*args, **kwargs)
        res = ana_obj.run(self._populations[population][0])
        self._add_result(res, population, str(ana_obj))
        return res

    def to_h5(self, fn, group_name=None, prefix=None):
        """
        hdf5 layout:
        prefix/group_name:
                          /full_matrix <-- Using base class serialization, represents full populations
                          /parameters <-- Implemented here
                          /subpopulations  <-- defined subpopulations. Using their own serialization. Descr. in attrs
                                         /population1  <-- using the registered names of defined subpopulations
                                         /population2
                                         /...
        """
        import h5py

        if prefix is None:
            prefix = "connectivity"
        if group_name is None:
            group_name = "topology"
        full_prefix = prefix + "/" + group_name

        super(TopologicalAnalysis, self).to_h5(fn, group_name="full_matrix", prefix=full_prefix)
        self._results.to_hdf(fn, key=full_prefix + "/" + _str_params)
        for pop_name, population in self._populations.items():
            if pop_name != 'all':  # We should already have this in "full_matrix".
                population[0].to_h5(fn, prefix=full_prefix + "/" + _str_subpop, group_name=pop_name)
        with h5py.File(fn, "a") as h5:
            data_grp = h5[full_prefix]
            data_grp.attrs["NEUROTOP_CLASS"] = "TopologicalAnalysis"
            subpop_grp = data_grp[_str_subpop]
            for pop_name, population in self._populations.items():
                if pop_name != 'all':
                    subpop_grp.attrs["__" + pop_name] = population[1]

    @classmethod
    def from_h5(cls, fn, group_name=None, prefix=None):
        import h5py

        if prefix is None:
            prefix = "connectivity"
        if group_name is None:
            group_name = "topology"
        full_prefix = prefix + "/" + group_name
        subpop_prefix = full_prefix + "/" + _str_subpop

        descr_dict = {}
        class_dict = {}
        with h5py.File(fn, "r") as h5:
            assert h5[full_prefix].attrs.get("NEUROTOP_CLASS", None) == cls.__name__
            subpop_grp = h5[subpop_prefix]
            for k, v in subpop_grp.attrs.items():
                if k.startswith("__"):
                    k = k[2:]
                    descr_dict[k] = v
                    class_dict[k] = subpop_grp[k].attrs.get("NEUROTOP_CLASS", None)

        base_obj = ConnectivityMatrix.from_h5(fn, group_name="full_matrix", prefix=full_prefix)
        base_obj = cls(base_obj._edges, vertex_properties=base_obj._vertex_properties,
                       default_edge_property=base_obj._default_edge, shape=base_obj._shape)
        base_obj._results = pandas.read_hdf(fn, key=full_prefix + "/" + _str_params)
        for subpop_name, subpop_cls in class_dict.items():
            if subpop_cls == cls.__name__:
                subpop_obj = TopologicalAnalysis.from_h5(fn, group_name=subpop_name, prefix=subpop_prefix)
            elif subpop_cls == "ConnectivityMatrix":
                subpop_obj = ConnectivityMatrix.from_h5(fn, group_name=subpop_name, prefix=subpop_prefix)
            else:
                raise RuntimeError("File {0} defines subpopulation of unknown type {1}".format(fn, subpop_cls))
            base_obj._populations[subpop_name] = (subpop_obj, descr_dict[subpop_name])
        return base_obj




