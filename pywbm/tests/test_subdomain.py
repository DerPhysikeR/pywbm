#!/usr/bin/env python
"""
2017-05-14 19:23:58
@author: Paul Reiter
"""
import pytest
import numpy as np
from numpy.testing import assert_allclose
from pywbm import Subdomain


def test_invalid_element():
    nodes = [(0, 0), (1, 0), (0, 1)]
    elements = [(0, 1), (1, 2), (2, 3)]
    with pytest.raises(ValueError):
        Subdomain(nodes, elements, 4*['i'],
                  4*[lambda x, y, z, k: 0])


def test_subdomain_normals():
    nodes = [(0, 0), (2, 0), (2, 2), (0, 2)]
    elements = [(0, 1), (1, 2), (2, 3), (3, 0)]
    sd = Subdomain(nodes, elements, 4*['i'],
                   4*[lambda x, y, z, k: 0])
    assert np.array_equal(
        np.array([(0, 1), (-1, 0), (0, -1), (1, 0)]),
        sd.normals)


def test_lx_ly():
    nodes = [(0, 0), (2, 0), (2, 2), (0, 2)]
    elements = [(0, 1), (1, 2), (2, 3), (3, 0)]
    sd = Subdomain(nodes, elements, 4*['i'],
                   4*[lambda x, y, z, k: 0])
    assert sd.lx == 2
    assert sd.ly == 2


def reference_pressure(a, b, k, x):
    return a*np.exp(1j*k*x) + b*np.exp(-1j*k*x)


@pytest.mark.parametrize('frequency', [100, 300, 800, 1000])
def test_velocity_bc(frequency):
    z = 1.205*343.4
    k = 2*np.pi*frequency/343.4
    lx, ly, n = 2, 1, 20
    nodes = [(0, 0), (lx, 0), (lx, ly), (0, ly)]
    elements = [(0, 1), (1, 2), (2, 3), (3, 0)]
    kinds = ['v', 'v', 'v', 'v']
    functions = 3*[lambda *args: 0] + [lambda *args: 1/z]
    sd = Subdomain(nodes, elements, kinds, functions)

    field_points = [(x, ly/2) for x in np.linspace(0, lx, 10)]
    solution = [sd.field_solution(x, y, z, k, n) for x, y in field_points]

    a = 1/(np.exp(2*1j*k*lx) - 1)
    b = 1 + a
    reference_solution = [reference_pressure(a, b, k, x)
                          for x, y in field_points]

    assert_allclose(np.array(reference_solution),
                    np.array(solution))


@pytest.mark.parametrize('frequency', [100, 300, 800, 1000])
def test_pressure_bc(frequency):
    z = 1.205*343.4
    k = 2*np.pi*frequency/343.4
    lx, ly, n = 2, 1, 20
    nodes = [(0, 0), (lx, 0), (lx, ly), (0, ly)]
    elements = [(0, 1), (1, 2), (2, 3), (3, 0)]
    kinds = ['v', 'v', 'v', 'p']
    functions = 3*[lambda *args: 0] + [lambda *args: 1]
    sd = Subdomain(nodes, elements, kinds, functions)

    field_points = [(x, ly/2) for x in np.linspace(0, lx, 10)]
    solution = [sd.field_solution(x, y, z, k, n) for x, y in field_points]

    a = 1/(np.exp(2*1j*k*lx) + 1)
    b = 1/(np.exp(-2*1j*k*lx) + 1)
    reference_solution = [reference_pressure(a, b, k, x)
                          for x, y in field_points]

    assert_allclose(np.array(reference_solution),
                    np.array(solution))


@pytest.mark.parametrize('frequency', [100, 300, 800, 1000])
def test_impedance_bc(frequency):
    z = 1.205*343.4
    k = 2*np.pi*frequency/343.4
    lx, ly, n = 2, 1, 20
    nodes = [(0, 0), (lx, 0), (lx, ly), (0, ly)]
    elements = [(0, 1), (1, 2), (2, 3), (3, 0)]
    kinds = ['v', 'p', 'v', 'z']
    functions = [lambda *args: 0, lambda *args: 1,
                 lambda *args: 0, lambda *args: 2*z]
    sd = Subdomain(nodes, elements, kinds, functions)

    field_points = [(x, ly/2) for x in np.linspace(0, lx, 10)]
    solution = [sd.field_solution(x, y, z, k, n) for x, y in field_points]

    # for impedance Z = 2*rho*c
    g = 1/3
    b = 1/(g*np.exp(1j*k*lx) + np.exp(-1j*k*lx))
    a = (1-b*np.exp(-1j*k*lx))/np.exp(1j*k*lx)
    reference_solution = [reference_pressure(a, b, k, x)
                          for x, y in field_points]

    assert_allclose(np.array(reference_solution),
                    np.array(solution))


@pytest.mark.parametrize('frequency', [100, 300, 800, 1000])
def test_interface_bc(frequency):
    # interface condition should act identical to impedance condition if no
    # other domain exists
    z = 1.205*343.4
    k = 2*np.pi*frequency/343.4
    lx, ly, n = 2, 1, 20
    nodes = [(0, 0), (lx, 0), (lx, ly), (0, ly)]
    elements = [(0, 1), (1, 2), (2, 3), (3, 0)]
    kinds = ['v', 'p', 'v', 'i']
    functions = [lambda *args: 0, lambda *args: 1,
                 lambda *args: 0, lambda *args: 2*z]
    sd = Subdomain(nodes, elements, kinds, functions)

    field_points = [(x, ly/2) for x in np.linspace(0, lx, 10)]
    solution = [sd.field_solution(x, y, z, k, n) for x, y in field_points]

    # for impedance Z = 2*rho*c
    g = 1/3
    b = 1/(g*np.exp(1j*k*lx) + np.exp(-1j*k*lx))
    a = (1-b*np.exp(-1j*k*lx))/np.exp(1j*k*lx)
    reference_solution = [reference_pressure(a, b, k, x)
                          for x, y in field_points]

    assert_allclose(np.array(reference_solution),
                    np.array(solution))


@pytest.mark.parametrize('x, y, inside', [
    (.5, .5, True),
    (1, 1, True),
    (2, 2, False)
])
def test_point_inside(x, y, inside):
    sd = Subdomain([(0, 0), (0, 1), (1, 1), (1, 0)],
                   [(0, 1), (1, 2), (2, 3), (3, 0)], None, None)
    assert sd.point_inside(x, y) == inside
