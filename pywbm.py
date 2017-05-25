#!/usr/bin/env python
"""
2017-05-13 21:05:35
@author: Paul Reiter
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hankel2
from pywbm import Subdomain


def vn(x, y, z, k):
    # incident velocity on left side
    # return (x == 0).astype(complex)*1j/(k*z)

    # incident velocity on left side
    return (x == 0).astype(complex)*0

    # sin-shaped velocity distribution on the bottom
    # return (y == 0).astype(complex)*1j/(k*z)*(np.sin(np.pi*x/lx))

    # sin-shaped velocity distribution on the left
    # return (x == 0).astype(complex)*1j/(k*z)*(np.sin(np.pi*y/ly))

    # incident velocity on the bottom
    # return (y == 0).astype(complex)*1j/(k*z)

    # incident velocity on the left and the bottom
    # return np.logical_or((x == 0), (y == 0)).astype(complex)*1j/(k*z)


def zn(x, y, z, k):
    # impedance of k*z on the right side
    # return (x == 2).astype(complex)*z
    return np.ones_like(x)*z


def pp(x, y, k):
    r = np.sqrt((x-.5)**2 + (y-.5)**2)
    return hankel2(0, k*r)


def gpp(n, x, y, k):
    r = np.sqrt((x-.5)**2 + (y-.5)**2)
    return k*(n[0]*(x-.5)/r*(hankel2(-1, k*r)/2 - hankel2(1, k*r)/2) +
              n[1]*(y-.5)/r*(hankel2(-1, k*r)/2 - hankel2(1, k*r)/2))


if __name__ == '__main__':

    z = 1.205*343.4
    k = 2*np.pi*800/343.4
    lx, ly, n = 2, 1, 20
    nodes = [(0, 0), (lx, 0), (lx, ly), (0, ly)]
    elements = [(0, 1), (1, 2), (2, 3), (3, 0)]
    kinds = ['z', 'z', 'z', 'z']
    functions = [zn, zn, zn, zn]
    sd = Subdomain(nodes, elements, kinds, functions, [(pp, gpp)])

    sd.solve(z, k, n, vn)

    x, y = np.meshgrid(np.linspace(0, lx, 84), np.linspace(0, ly, 44))
    z = np.real(sd.field_solution(x, y, z, k, n, vn))
    plt.contourf(x, y, z)
    plt.colorbar()
    plt.show()
