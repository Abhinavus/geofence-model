import unittest
from unittest.mock import patch
from ..fence import Fence
import numpy as np


class FenceTest(unittest.TestCase):

    vs = [[0, 0], [1, 1], [2, 0]]

    def test_fence(self):
        fence = Fence(self.vs)
        self.assertIsInstance(Fence(), type(fence))
        self.assertRaises(AssertionError, lambda: Fence([[1, 2], [3, 4]]))
        # TODO: total length count, must be even
        # self.assertRaises(AssertionError, lambda: Fence([[1, 2], [3, 4], [5, 6], [7]]))

    def test_name(self):
        fence = Fence()
        fence.site = "io"
        self.assertEqual("io", fence.site)  # test default name

    def test_get_vertices(self):
        fence = Fence(self.vs)
        self.assertTrue(np.array_equal(fence.vertices, self.vs))

    def test_set_vertices(self):
        fence = Fence(self.vs)
        vs2 = np.array([1, 1, 2, 0, 0, 0])
        fence.vertices = vs2
        self.assertTrue(np.array_equal(fence.vertices, vs2))

    @patch('geofencing.fence._import_file', return_value=['name', 'someList'])
    def test_vertices_from_file(self, mock_import):
        fence = Fence()
        self.assertEqual(None, fence.set_vertices_from_file('/path'))
        self.assertEqual('name', fence.site)
        self.assertEqual('someList', fence.vertices)

    @patch('geofencing.fence._point_in_poly', return_value=None)
    def test_detect(self, mock_poly):
        fence = Fence(self.vs)
        point1 = [1, 0.5]
        self.assertEqual(None, fence.detect(point1))
        self.assertEqual(None, fence.detect(point1, 'rc'))
        self.assertEqual(None, fence.detect(point1, 'wn'))
        self.assertEqual(None, fence.detect(point1, 'rc_vectorize'))
        self.assertEqual(None, fence.detect(point1, 'wn_vectorize'))
        self.assertEqual(None, fence.detect(point1, 'wn_edge'))

        point2 = [1]
        self.assertRaises(AssertionError, lambda: fence.detect(point2))
        self.assertRaises(AssertionError, lambda: fence.detect(point2, 'rc'))
        self.assertRaises(AssertionError, lambda: fence.detect(point2, 'wn'))
        self.assertRaises(AssertionError, lambda: fence.detect(point2, 'rc_vectorize'))
        self.assertRaises(AssertionError, lambda: fence.detect(point2, 'wn_vectorize'))
        self.assertRaises(AssertionError, lambda: fence.detect(point2, 'wn_edge'))
