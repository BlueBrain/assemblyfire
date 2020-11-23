from ..assemblies import AssemblyGroup


class NetworkAssemblyGroup(AssemblyGroup):
    """
    A class derived from AssemblyGroup with additional information on the connectivity within the assembly
    """
    # TODO: Does it make sense to derive this from AssemblyGroup?
    #  Maybe just make this a stand-alone thing that simply applies topological functions on Assemblies
    # TODO: Implement actual topological functionality?
    def __init__(self, assemblies, all_gids, connectivity_obj, label=None, metadata=None):
        self._connectivity = connectivity_obj
        super(NetworkAssemblyGroup, self).__init__(assemblies, all_gids, label=label, metadata=metadata)

    @classmethod
    def attach_connectivity(cls, base_group, connectivity_obj):
        return cls(base_group.assemblies, base_group.all, connectivity_obj,
                   label=base_group.label, metadata=base_group.metadata)

    def mat_of(self, idx):
        """
        Returns the connection matrix of a contained assembly
        :return:
        """
        return self._connectivity.submat(self.iloc(idx))

    def __mul__(self, other):
        base_spec = super(NetworkAssemblyGroup, self).__mul__(other)
        return self.attach_connectivity(base_spec, self._connectivity)

    def __add__(self, other):
        base_spec = super(NetworkAssemblyGroup, self).__add__(other)
        return self.attach_connectivity(base_spec, self._connectivity)

    def to_h5(self, filename, prefix=None, version=None):
        # TODO: Save the connectivity matrix somewhere in the metadata at 'prefix'
        super(NetworkAssemblyGroup, self).to_h5(filename, prefix=prefix, version=version)
