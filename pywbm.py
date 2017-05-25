#!/usr/bin/env python
"""
2017-05-13 21:05:35
@author: Paul Reiter
"""
import numpy as np
import matplotlib.pyplot as plt
from pywbm import Subdomain

z = 1.205*343.4
k = 2*np.pi*800/343.4
lx, ly, n = 2, 1, 100
nodes = [(0, 0), (lx, 0), (lx, ly), (0, ly)]
elements = [(0, 1), (1, 2), (2, 3), (3, 0)]
sd = Subdomain(nodes, elements)


def vn(x, y):
    # incident velocity on left side
    return (x == 0).astype(complex)*1j/(k*z)

    # sin-shaped velocity distribution on the bottom
    # return (y == 0).astype(complex)*1j/(k*z)*(np.sin(np.pi*x/lx))

    # sin-shaped velocity distribution on the left
    # return (x == 0).astype(complex)*1j/(k*z)*(np.sin(np.pi*y/ly))

    # incident velocity on the bottom
    # return (y == 0).astype(complex)*1j/(k*z)

    # incident velocity on the left and the bottom
    # return np.logical_or((x == 0), (y == 0)).astype(complex)*1j/(k*z)


sd.solve(z, k, n, vn)

x, y = np.meshgrid(np.linspace(0, lx, 84), np.linspace(0, ly, 44))
z = sd.field_solution(x, y, z, k, n, vn)
plt.contourf(x, y, z)
plt.show()
