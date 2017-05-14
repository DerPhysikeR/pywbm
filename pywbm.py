#!/usr/bin/env python
"""
2017-05-13 21:05:35
@author: Paul Reiter
"""
from functools import partial
import numpy as np
from scipy.integrate import fixed_quad
import matplotlib.pyplot as plt


def complex_quad(function, a, b, n):
    def real_function(t):
        return np.real(function(t))

    def imag_function(t):
        return np.imag(function(t))

    real_part = fixed_quad(real_function, a, b, n=n)
    imag_part = fixed_quad(imag_function, a, b, n=n)

    return real_part[0] + 1j*imag_part[0]


def phi_w_r(kxwr, kywr, x, y):
    return np.cos(kxwr*x)*np.exp(-1j*kywr*y)


def grad_phi_w_r(kxwr, kywr, nx, ny, x, y):
    return (-nx*kxwr*np.sin(kxwr*x)*np.exp(-1j*kywr*y) +
            -ny*1j*kywr*np.cos(kxwr*x)*np.exp(-1j*kywr*y))


def phi_w_s(kxws, kyws, x, y):
    return np.exp(-1j*kxws*x)*np.cos(kyws*y)


def grad_phi_w_s(kxws, kyws, nx, ny, x, y):
    return (-nx*1j*kxws*np.exp(-1j*kxws*x)*np.cos(kyws*y) +
            -ny*kyws*np.exp(-1j*kxws*x)*np.sin(kyws*y))


def kxwr_kywr(lx, k):
    kxwr, result = 0, []
    while kxwr < k:
        result.append((kxwr, -np.sqrt(k**2 - kxwr**2)))
        result.append((kxwr, +np.sqrt(k**2 - kxwr**2)))
        kxwr += np.pi/lx
    return result


def kxws_kyws(ly, k):
    kyws, result = 0, []
    while kyws < k:
        result.append((-np.sqrt(k**2 - kyws**2), kyws))
        result.append((+np.sqrt(k**2 - kyws**2), kyws))
        kyws += np.pi/ly
    return result


def line_integral(function, x0, y0, x1, y1, n):

    def to_quad(t):
        return function(x0 + t*(x1-x0), y0 + t*(y1-y0))

    return complex_quad(to_quad, 0, 1, n)


def integrate_square(function, x0, y0, x1, y1, n):
    """`function` expects x, y coordinates and a normal vector nx, ny
    function(nx, ny, x, y)"""
    if x1 <= x0 or y1 <= y0:
        raise ValueError('Not the corners of a square!')

    return sum([line_integral(partial(function, 0, 1), x0, y0, x1, y0, n),
                line_integral(partial(function, -1, 0), x1, y0, x1, y1, n),
                line_integral(partial(function, 0, -1), x1, y1, x0, y1, n),
                line_integral(partial(function, 1, 0), x0, y1, x0, y0, n)])


def get_av(pw, gpw, z, k):
    def av(nx, ny, x, y):
        return 1j/(z*k)*pw(x, y)*gpw(nx, ny, x, y)
    return av


def a_ij(z, k, phiw, grad_phiw, x0, y0, x1, y1, n):
    a = np.empty((len(phiw), len(phiw)), dtype=complex)
    for i, pw in enumerate(phiw):
        for j, gpw in enumerate(grad_phiw):
            a[i, j] = integrate_square(get_av(pw, gpw, z, k),
                                       x0, y0, x1, y1, n)
    return a


def get_rhs(pw, vn):
    def fv(nx, ny, x, y):
        return pw(x, y)*vn(x, y)
    return fv


def rhs_i(z, k, phiw, vn, x0, y0, x1, y1, n):
    rhs = np.empty(len(phiw), dtype=complex)
    for i, pw in enumerate(phiw):
        rhs[i] = integrate_square(get_rhs(pw, vn),
                                  x0, y0, x1, y1, n)
    return rhs


if __name__ == '__main__':
    z = 1.205*343.4
    k = 2*np.pi*800/343.4
    lx, ly, n = 2, 1, 100

    kwr, kws = kxwr_kywr(lx, k), kxws_kyws(ly, k)
    phiw = [partial(phi_w_r, *kw) for kw in kwr] + \
           [partial(phi_w_s, *kw) for kw in kws]
    grad_phiw = [partial(grad_phi_w_r, *kw) for kw in kwr] + \
                [partial(grad_phi_w_s, *kw) for kw in kws]

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

    a = a_ij(z, k, phiw, grad_phiw, 0, 0, lx, ly, n)
    rhs = rhs_i(z, k, phiw, vn, 0, 0, lx, ly, n)
    pw = np.linalg.solve(a, rhs)

    def solution(x, y):
        return sum(p[0]*p[1](x, y) for p in zip(pw, phiw))

    x, y = np.meshgrid(np.linspace(0, 2, 84), np.linspace(0, 1, 44))
    z = solution(x, y)
    plt.contourf(x, y, z)
    plt.show()
