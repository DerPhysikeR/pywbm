#!/usr/bin/env python
"""
2017-08-30 20:26:25
@author: Paul Reiter
"""
import pytest
import numpy as np
from numpy.testing import assert_allclose
from pywbm import Subdomain, Multidomain
from test_subdomain import reference_pressure


@pytest.mark.parametrize('frequency', [100, 300, 800, 1000])
def test_two_domains_diagonal_interface(frequency):
    z = 1.205*343.4
    k = 2*np.pi*frequency/343.4
    lx, ly, n = 2, 1, 20

    # subdomain 1
    nodes = [(0, 0), (lx/2, 0), (lx/3, ly), (0, ly)]
    elements = [(0, 1), (1, 2), (2, 3), (3, 0)]
    kinds = ['v', 'i', 'v', 'v']
    functions = [lambda *args: 0, lambda *args: z,
                 lambda *args: 0, lambda *args: 1/z]
    sd1 = Subdomain(nodes, elements, kinds, functions)

    # subdomain 2
    nodes = [(lx/2, 0), (lx, 0), (lx, ly), (lx/3, ly)]
    elements = [(0, 1), (1, 2), (2, 3), (3, 0)]
    kinds = ['v', 'v', 'v', 'i']
    functions = [lambda *args: 0, lambda *args: 0,
                 lambda *args: 0, lambda *args: z]
    sd2 = Subdomain(nodes, elements, kinds, functions)

    # create multidomain
    md = Multidomain([sd1, sd2])

    # check
    field_points = [(x, ly/2) for x in np.linspace(0, lx, 10)]
    solution = [md.field_solution(x, y, z, k, n) for x, y in field_points]

    a = 1/(np.exp(2*1j*k*lx) - 1)
    b = 1 + a
    reference_solution = [reference_pressure(a, b, k, x)
                          for x, y in field_points]

    assert_allclose(np.array(reference_solution),
                    np.array(solution), rtol=1e-5)
