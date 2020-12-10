import pandas

import analysis_implementations
from .connectivity import ConnectivityMatrix


class TopologicalAnalysis(ConnectivityMatrix):
    def __init__(self, *args, **kwargs):
        super(TopologicalAnalysis, self).__init__(*args, **kwargs)
        self._populations = {
            'all': (self, "Default population describing the entire population")
        }
        idx = pandas.MultiIndex.from_tuples([], names=['population', 'analysis'])
        self._results = pandas.Series([], index=idx, dtype=object)

    @property
    def results(self):
        return self._results

    @property
    def populations(self):
        return self._populations

    def _add_result(self, res, population, label):
        idx_tuples = self._results.index.to_numpy().tolist()
        idx_tuples.append((population, label))
        idx = pandas.MultiIndex.from_tuples(idx_tuples, names=self._results.index.names)

        old_res = self._results.values.tolist()
        old_res.append(res)
        self._results = pandas.Series(old_res, index=idx)

    def name_population(self, population, label, description=None):
        if description is None:
            description = "A custom population"
        self._populations[label] = (population, description)

    def run(self, analysis, population, *args, **kwargs):
        #  TODO: Test if it is already int he results
        if analysis not in analysis_implementations.__dict__:
            raise ValueError("Analysis {0} unknown!".format(analysis))
        if population not in self._populations:
            raise ValueError("Population {0} undefined!".format(population))
        func = analysis_implementations.__dict__[analysis]
        label = kwargs.pop("analysis_label", analysis)
        res = func.run(self._populations[population][0], *args, **kwargs)
        self._add_result(res, population, label)
        return res



