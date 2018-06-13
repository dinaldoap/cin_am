class Sample:
    """
    Representation of a sample
    """

    def __init__(self, label, new_label, data):
        self.label = label
        self.new_label = new_label
        self.data = data
        self.original_data = list(data)


class Prototype:
    """
    Representation of a prototype
    """

    def __init__(self, label, data):
        self.label = label
        self.data = data


class Cluster:
    """
    Representation of a cluster
    """

    def __init__(self):
        self.elements = list()

    def append(self, e):
        self.elements.append(e)

    def size(self):
        return len(self.elements)


class View:
    """
    Representation of a view
    """

    def __init__(self, name, cols, labelCol, startRow, ignoreCols=[]):
        self.name = name
        self._cols = list(cols)
        self.labelCol = labelCol
        self.startRow = startRow
        self._ignoreCols = ignoreCols

    def cols(self):
        return [c for c in self._cols if c not in self._ignoreCols]


class ExecutionResult:
    """
    Representation of an Execution Result
    """

    def __init__(self, jkcm_k_gh, rai, clusters, g, _1_s2, iterations_convergence):
        self.jkcm_k_gh = jkcm_k_gh
        self.rai = rai
        self.clusters = clusters
        self.g = g
        self._1_s2 = _1_s2
        self.iterations_convergence = iterations_convergence