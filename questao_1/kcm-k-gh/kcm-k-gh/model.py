class Sample:
    """
    Representation of a sample
    """

    def __init__(self, label, new_label, data):
        self.label = label
        self.new_label = new_label
        self.data = data


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
