#!/usr/bin/env python
"""
2017-05-25 12:21:09
@author: Paul Reiter
"""
from functools import partial
import numpy as np
from scipy.integrate import fixed_quad


def line_integral(function, p0, p1, n):
    x0, y0 = p0[0], p0[1]
    x1, y1 = p1[0], p1[1]

    def to_quad(t):
        return function(x0 + t*(x1-x0), y0 + t*(y1-y0))

    return fixed_quad(to_quad, 0, 1, n=n)[0]


class Wavefunctions():

    def __init__(self, k, lx, ly):
        if k <= 0:
            raise ValueError('Invalid wave number!')
        if lx <= 0 or ly <= 0:
            raise ValueError('Invalid dimensions lx, ly!')

        kwr, kws = self.kxwr_kywr(lx, k), self.kxws_kyws(ly, k)
        self.phiw = [partial(self.phi_w_r, *kw) for kw in kwr] + \
                    [partial(self.phi_w_s, *kw) for kw in kws]
        self.grad_phiw = [partial(self.grad_phi_w_r, *kw) for kw in kwr] + \
                         [partial(self.grad_phi_w_s, *kw) for kw in kws]

    def phi_w_r(self, kxwr, kywr, x, y):
        return np.cos(kxwr*x)*np.exp(-1j*kywr*y)

    def grad_phi_w_r(self, kxwr, kywr, nx, ny, x, y):
        return (-nx*kxwr*np.sin(kxwr*x)*np.exp(-1j*kywr*y) +
                -ny*1j*kywr*np.cos(kxwr*x)*np.exp(-1j*kywr*y))

    def phi_w_s(self, kxws, kyws, x, y):
        return np.exp(-1j*kxws*x)*np.cos(kyws*y)

    def grad_phi_w_s(self, kxws, kyws, nx, ny, x, y):
        return (-nx*1j*kxws*np.exp(-1j*kxws*x)*np.cos(kyws*y) +
                -ny*kyws*np.exp(-1j*kxws*x)*np.sin(kyws*y))

    def kxwr_kywr(self, lx, k):
        kxwr, result = 0, []
        while kxwr < k:
            result.append((kxwr, -np.sqrt(k**2 - kxwr**2)))
            result.append((kxwr, +np.sqrt(k**2 - kxwr**2)))
            kxwr += np.pi/lx
        return result

    def kxws_kyws(self, ly, k):
        kyws, result = 0, []
        while kyws < k:
            result.append((-np.sqrt(k**2 - kyws**2), kyws))
            result.append((+np.sqrt(k**2 - kyws**2), kyws))
            kyws += np.pi/ly
        return result
