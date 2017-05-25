#!/usr/bin/env python
"""
2017-05-14 18:35:02
@author: Paul Reiter
"""
import numpy as np


class Subdomain():

    def __init__(self, nodes, elements, bounded=True):
        if max([max(ele) for ele in elements]) > len(nodes)-1:
            raise ValueError('Element references non-existing node!')
        self.bounded = bounded
        self.nodes = np.array(nodes)
        self.elements = np.array(elements)
        self.lx = np.max(self.nodes[:, 0]) - np.min(self.nodes[:, 0])
        self.ly = np.max(self.nodes[:, 1]) - np.min(self.nodes[:, 1])

    @property
    def normals(self):
        normals = np.empty_like(self.elements)
        for i, element in enumerate(self.elements):
            p0, p1 = self.nodes[element[0]], self.nodes[element[1]]
            length = np.sqrt((p1[0] - p0[0])**2 + (p1[1] - p0[1])**2)
            normals[i, 0] = (p0[1]-p1[1]) / length
            normals[i, 1] = (p1[0]-p0[0]) / length
        return normals

    def get_subdomain_matrix(self, z, k):
        pass

    def get_subdomain_rhs(self, z, k):
        pass

    def add_source(self):
        pass


# class BoundedSubdomain(Subdomain):
#     pass

# class UnboundedSubdomain(Subdomain):
#     pass
