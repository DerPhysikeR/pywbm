#!/usr/bin/env python
"""
2017-05-14 18:35:02
@author: Paul Reiter
"""
from collections import OrderedDict
from functools import partial
import numpy as np
from .wavefunctions import Wavefunctions, line_integral


class Subdomain():

    def __init__(self, nodes, elements, bounded=True):
        if max([max(ele) for ele in elements]) > len(nodes)-1:
            raise ValueError('Element references non-existing node!')
        self.bounded = bounded
        self.nodes = np.array(nodes)
        self.elements = np.array(elements)
        self.lx = np.max(self.nodes[:, 0]) - np.min(self.nodes[:, 0])
        self.ly = np.max(self.nodes[:, 1]) - np.min(self.nodes[:, 1])
        self.solutions = OrderedDict()

    @property
    def normals(self):
        normals = np.empty_like(self.elements)
        for i, element in enumerate(self.elements):
            p0, p1 = self.nodes[element[0]], self.nodes[element[1]]
            length = np.sqrt((p1[0] - p0[0])**2 + (p1[1] - p0[1])**2)
            normals[i, 0] = (p0[1]-p1[1]) / length
            normals[i, 1] = (p1[0]-p0[0]) / length
        return normals

    def solve(self, z, k, n, vn):
        if (z, k, n, vn) not in self.solutions:
            wv = Wavefunctions(k, self.lx, self.ly)
            a = self.a_ij(z, k, wv.phiw, wv.grad_phiw, n)
            rhs = self.rhs_i(z, k, wv.phiw, vn, n)
        self.solutions[(z, k, n, vn)] = np.linalg.solve(a, rhs)

    def get_av(self, pw, gpw, z, k):
        def av(normal, x, y):
            return 1j/(z*k)*pw(x, y)*gpw(normal[0], normal[1], x, y)
        return av

    def get_rhs(self, pw, vn):
        def fv(normal, x, y):
            return pw(x, y)*vn(x, y)
        return fv

    def a_ij(self, z, k, phiw, grad_phiw, n):
        a = np.zeros((len(phiw), len(phiw)), dtype=complex)
        for i, pw in enumerate(phiw):
            for j, gpw in enumerate(grad_phiw):
                for normal, element in zip(self.normals, self.elements):
                    p0, p1 = self.nodes[element[0]], self.nodes[element[1]]
                    fun = partial(self.get_av(pw, gpw, z, k), normal)
                    a[i, j] += line_integral(fun, p0, p1, n)
        return a

    def rhs_i(self, z, k, phiw, vn, n):
        rhs = np.zeros(len(phiw), dtype=complex)
        for i, pw in enumerate(phiw):
            for normal, element in zip(self.normals, self.elements):
                p0, p1 = self.nodes[element[0]], self.nodes[element[1]]
                fun = partial(self.get_rhs(pw, vn), normal)
                rhs[i] += line_integral(fun, p0, p1, n)
        return rhs

    def field_solution(self, x, y, z, k, n, vn):
        if (z, k, n, vn) not in self.solutions:
            self.solve(z, k, n, vn)

        pw = self.solutions[(z, k, n, vn)]
        wv = Wavefunctions(k, self.lx, self.ly)
        return sum(p[0]*p[1](x, y) for p in zip(pw, wv.phiw))
