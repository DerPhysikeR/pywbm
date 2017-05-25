#!/usr/bin/env python
"""
2017-05-14 19:23:58
@author: Paul Reiter
"""
import pytest
import numpy as np
from pywbm import Subdomain


def test_invalid_element():
    nodes = [(0, 0), (1, 0), (0, 1)]
    elements = [(0, 1), (1, 2), (2, 3)]
    with pytest.raises(ValueError):
        Subdomain(nodes, elements)


def test_subdomain_normals():
    nodes = [(0, 0), (2, 0), (2, 2), (0, 2)]
    elements = [(0, 1), (1, 2), (2, 3), (3, 0)]
    sd = Subdomain(nodes, elements)
    assert np.array_equal(
        np.array([(0, 1), (-1, 0), (0, -1), (1, 0)]),
        sd.normals)
