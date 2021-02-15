import numpy
from assemblyfire import TopologicalAnalysis

m = numpy.random.rand(100, 100)
m2 = numpy.random.rand(100, 100)
m[m > 0.05] = 0
m[numpy.eye(m.shape[0]) == 1] = 0
m2[m == 0] = 0

C = TopologicalAnalysis(m, vertex_properties={'a': numpy.arange(100), 'b': numpy.arange(5, 105)}, edge_properties={'alternative': m2})
C.name_population(C.index('a').lt(75), "a under 75")
C.name_population(C.filter('alternative').gt(0.5), "alternative over 0.5")
C.name_population(C.default("alternative"), "with alternative edge weight")

C.run("Degree", "all", kind="out")
C.run("Degree", "all", kind="in")
C.run("Degree", "with alternative edge weight")
C.run("Degree", "alternative over 0.5")

raw_result = C.run("Bettis", "some population")
area_under_curve(raw_result)

C.to_h5("deleteme.h5")

C = TopologicalAnalysis.from_h5("deletame.h5")
raw_result = C.results[("Bettis", "some population")]

area_under_curve(raw_result)

