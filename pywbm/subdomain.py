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

    def __init__(self, nodes, elements, kinds, functions, sources=None,
                 bounded=True, cache_length=np.inf):
        """Creates a single `Subdomain` for the wave based method.

        Parameters
        ----------
        nodes : array-like
            List or array of coordinate tuples
        elements : array-like
            List or array of node-number tuples
        kinds : array-like
            List or array of characters. Only 'v', 'z', 'p', 'i' are valid.
            Marking velocity, impedance, pressure and interface boundary
            conditions.
        functions : list
            List of functions f(x, y, z, k) for the quantities given by `kinds`
        sources : list
            List of source function tuples [(p1, grap1), (p2, gradp2), ...]
            p1(k, x, y), p2(k, (nx, ny), x, y)
        bounded : boolean
            Optional, marks boundary as bounded as opposed to open
            Default is True, which means bounded.
        cache_length : integer
            Optional, number of solutions saved if `solve` is called multiple
            times with different parameters
        """
        if max([max(ele) for ele in elements]) > len(nodes)-1:
            raise ValueError('Element references non-existing node!')
        self.bounded = bounded
        self.nodes = np.array(nodes)
        self.elements = np.array(elements)
        self.kinds = kinds
        self.functions = functions
        self.lx = np.max(self.nodes[:, 0]) - np.min(self.nodes[:, 0])
        self.ly = np.max(self.nodes[:, 1]) - np.min(self.nodes[:, 1])
        self.solutions = OrderedDict()
        self.cache_length = cache_length

        if sources is not None:
            def source(x, y, k):
                return sum([s[0](x, y, k) for s in sources])

            def grad_source(n, x, y, k):
                return sum([s[1](n, x, y, k) for s in sources])

            self.pp = source
            self.gpp = grad_source
        else:
            self.pp = lambda x, y, k: 0
            self.gpp = lambda n, x, y, k: 0

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
            a = self.a_ij(z, k, n)
            rhs = self.rhs_i(z, k, vn, n)
        if (len(self.solutions) >= self.cache_length and
           (z, k, n, vn) not in self.solutions):
            self.solutions.popitem(last=False)
        self.solutions[(z, k, n, vn)] = np.linalg.solve(a, rhs)

    def get_a(self, pwt, gpwt, pw, gpw, z, k, kind, fun):
        if kind == 'v':
            def afun(n, x, y):
                return 1j/(z*k)*pwt(x, y)*gpw(n[0], n[1], x, y)
        elif kind == 'z':
            def afun(n, x, y):
                return (1j/(z*k)*pwt(x, y)*gpw(n[0], n[1], x, y) -
                        1/fun(x, y, z, k)*pwt(x, y)*pw(x, y))
        elif kind == 'p':
            def afun(n, x, y):
                return -1j/(z*k)*gpwt(n[0], n[1], x, y)*pw(x, y)
        elif kind == 'i':
            def afun(n, x, y):
                return (1j/(z*k)*pwt(x, y)*gpw(n[0], n[1], x, y) -
                        1/fun(x, y, z, k)*pwt(x, y)*pw(x, y))
        else:
            raise ValueError('Only kinds v, z, p and i are allowed for'
                             ' element!')
        return afun

    def get_rhs(self, pwt, gpwt, z, k, kind, fun):
        if kind == 'v':
            def ffun(n, x, y):
                return pwt(x, y)*(fun(x, y, z, k) -
                                  1j/(z*k)*self.gpp(n, x, y, k))
        elif kind == 'z':
            def ffun(n, x, y):
                return pwt(x, y)*(self.pp(x, y, k)/fun(x, y, z, k) -
                                  1j/(z*k)*self.gpp(n, x, y, k))
        elif kind == 'p':
            def ffun(n, x, y):
                return 1j/(z*k)*gpwt(n[0], n[1], x, y)*(self.pp(x, y, k) -
                                                        fun(x, y, z, k))
        elif kind == 'i':
            def ffun(n, x, y):
                return pwt(x, y)*(self.pp(x, y, k)/fun(x, y, z, k) -
                                  1j/(z*k)*self.gpp(n, x, y, k))
        else:
            raise ValueError('Only kinds v, z, p and i are allowed for'
                             ' element!')
        return ffun

    def a_ij(self, z, k, n):
        wv = Wavefunctions(k, self.lx, self.ly)
        a = np.zeros((len(wv.phiw), len(wv.phiw)), dtype=complex)
        # pwt ... t is for transposed
        for i, (pwt, gpwt) in enumerate(zip(wv.phiw, wv.grad_phiw)):
            for j, (pw, gpw) in enumerate(zip(wv.phiw, wv.grad_phiw)):
                for normal, element, kind, fun in zip(
                  self.normals, self.elements, self.kinds, self.functions):
                    p0, p1 = self.nodes[element[0]], self.nodes[element[1]]
                    function = partial(self.get_a(pwt, gpwt, pw, gpw, z, k,
                                                  kind, fun), normal)
                    a[i, j] += line_integral(function, p0, p1, n)
        return a

    def rhs_i(self, z, k, vn, n):
        wv = Wavefunctions(k, self.lx, self.ly)
        rhs = np.zeros(len(wv.phiw), dtype=complex)
        # pwt ... t is for transposed
        for i, (pwt, gpwt) in enumerate(zip(wv.phiw, wv.grad_phiw)):
            for normal, element, kind, fun in zip(
              self.normals, self.elements, self.kinds, self.functions):
                p0, p1 = self.nodes[element[0]], self.nodes[element[1]]
                function = partial(self.get_rhs(pwt, gpwt, z, k, kind, fun),
                                   normal)
                rhs[i] += line_integral(function, p0, p1, n)
        return rhs

    def field_solution(self, x, y, z, k, n, vn):
        if (z, k, n, vn) not in self.solutions:
            self.solve(z, k, n, vn)

        pw = self.solutions[(z, k, n, vn)]
        wv = Wavefunctions(k, self.lx, self.ly)
        homogeneous = sum(p[0]*p[1](x, y) for p in zip(pw, wv.phiw))
        particular = self.pp(x, y, k)
        return homogeneous + particular
