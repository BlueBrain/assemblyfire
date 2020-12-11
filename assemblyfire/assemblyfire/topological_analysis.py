import pandas

from assemblyfire import topology_implementations
from ._topological_analysis import AnalysisImplementation
from .connectivity import ConnectivityMatrix


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



