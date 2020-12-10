class Bettis():
    @classmethod
    def run(cls, conn):
        import pyflagser
        adj_mat = conn.matrix
        return pyflagser.flagser_unweighted(adj_mat, directed=True)["betti"]
