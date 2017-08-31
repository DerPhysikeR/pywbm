#!/usr/bin/env python
"""
2017-05-25 15:25:04
@author: Paul Reiter
"""
from collections import OrderedDict
from functools import partial
import numpy as np
# from .subdomain import Subdomain
from .wavefunctions import Wavefunctions, line_integral


class Multidomain():

    def __init__(self, subdomains, cache_length=np.inf):
        self.subdomains = subdomains
        self.solutions = OrderedDict()
        self.cache_length = cache_length

    def solve(self, z, k, n):
        if (z, k, n) not in self.solutions:
            capital_a_list, rhs_list = [], []
            for sub1 in self.subdomains:
                capital_a_line, rhs_line = [], []
                for sub2 in self.subdomains:
                    if sub1 is sub2:
                        capital_a_line.append(sub1.a_ij(z, k, n))
                        rhs_line.append(sub1.rhs_i(z, k, n))
                    else:
                        capital_a_line.append(self.c_ij(z, k, n, sub1, sub2))
                        rhs_line.append(self.rhs_i(z, k, n, sub1, sub2))
                capital_a_list.append(
                    np.hstack([np.array(a) for a in capital_a_line]))
                rhs_list.append(np.sum([np.array(rl) for rl in rhs_line], axis=0))
            capital_a = np.vstack(capital_a_list)
            rhs = np.concatenate(rhs_list)
            solution = np.linalg.solve(capital_a, rhs)
            index = 0
            for sub in self.subdomains:
                len_items = len(Wavefunctions(k, sub.lx, sub.ly).phiw)
                sub.solutions[(z, k, n)] = solution[index:index+len_items]
                index += len_items
            self.solutions[(z, k, n)] = 'DONE'

    def get_rhs(self, pwt, z, k, kind, fun, sub2):
        if kind == 'i':
            def ffun(n, x, y):
                return pwt(x, y)*(-sub2.pp(x, y, k)/fun(x, y, z, k) -
                                  1j/(z*k)*sub2.gpp(-n, x, y, k))
        elif kind in 'vzp':
            def ffun(n, x, y):
                return 0
        else:
            raise ValueError('Only kinds v, z, p and i are allowed for'
                             ' element!')
        return ffun

    def rhs_i(self, z, k, n, sub1, sub2):
        wv = Wavefunctions(k, sub1.lx, sub1.ly)
        rhs = np.zeros(len(wv.phiw), dtype=complex)
        for i, (pwt, gpwt) in enumerate(zip(wv.phiw, wv.grad_phiw)):
            for normal, element, kind, fun in zip(
              sub1.normals, sub1.elements, sub1.kinds, sub1.functions):
                p0, p1 = sub1.nodes[element[0]], sub1.nodes[element[1]]
                function = partial(self.get_rhs(pwt, z, k, kind, fun, sub2),
                                   normal)
                rhs[i] += line_integral(function, p0, p1, n)
        return rhs

    def get_c(self, pwt, gpwt, pw, gpw, z, k, kind, fun):
        if kind == 'i':
            def cfun(n, x, y):
                return (1j/(z*k)*pwt(x, y)*gpw(n[0], n[1], x, y) +
                        1/fun(x, y, z, k)*pwt(x, y)*pw(x, y))
        elif kind in 'vzp':
            def cfun(n, x, y):
                return 0
        else:
            raise ValueError('Only kinds v, z, p and i are allowed for'
                             ' element!')
        return cfun

    def c_ij(self, z, k, n, sub1, sub2):
        wv1 = Wavefunctions(k, sub1.lx, sub1.ly)
        wv2 = Wavefunctions(k, sub2.lx, sub2.ly)
        c = np.zeros((len(wv1.phiw), len(wv2.phiw)), dtype=complex)
        for i, (pwt, gpwt) in enumerate(zip(wv1.phiw, wv1.grad_phiw)):
            for j, (pw, gpw) in enumerate(zip(wv2.phiw, wv2.grad_phiw)):
                for normal, element, kind, fun in zip(
                  sub1.normals, sub1.elements, sub1.kinds, sub1.functions):
                    p0, p1 = sub1.nodes[element[0]], sub1.nodes[element[1]]
                    function = partial(self.get_c(pwt, gpwt, pw, gpw, z, k,
                                                  kind, fun), normal)
                    c[i, j] += line_integral(function, p0, p1, n)
        return c

    def field_solution(self, x, y, z, k, n):
        if (z, k, n) not in self.solutions:
            self.solve(z, k, n)
        for sub in self.subdomains:
            if sub.point_inside(x, y):
                return sub.field_solution(x, y, z, k, n)
        return np.nan
