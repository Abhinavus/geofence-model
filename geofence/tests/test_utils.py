import unittest
from ..utils import _is_left, _check_off_edge, _point_in_poly, _import_file, \
    _wn_point_in_poly, _wn_vectorize, _wn_edge,\
    _rc_point_in_poly, _rc_vectorize
import numpy as np


class UtilsTest(unittest.TestCase):
    # Convex U shape facing west
    vertices = np.array([[0, 0], [0, 2], [3, 2], [3, 0], [2, 0], [2, 1], [1, 1], [1, 0]])
    point1 = [1.5, 1.5]  # inside vertical area
    point2 = [1.5, 0.5]  # outside between horizontal areas
    point3 = [4, 2]      # outside above
    point4 = [2.5, 0.5]  # inside top horizontal area
    point5 = [0, 1]      # On edge
    point6 = [0, 2]      # On vertice

    def test_is_left(self):
        point = [0, 0]                 # test point, the origin

        vi1, vj1 = [0, 1], [1, 0]      # origin on the left
        self.assertTrue(_is_left(*point, *vi1, *vj1) > 0)

        vi2, vj2 = [0, -1], [1, 0]     # origin on the right
        self.assertTrue(_is_left(*point, *vi2, *vj2) < 0)

        vi3, vj3 = [1, 0], [-1, 0]     # origin on the line
        self.assertTrue(_is_left(*point, *vi3, *vj3) == 0)

    def test_check_edge(self):
        # Single edge case
        edge = np.array([[0, 0], [1, 1]])  # coordinate order [y, x]

        point1 = [0, 1]     # Off edge
        delta1 = edge - point1
        dot1 = delta1[0] @ delta1[1]
        cross1 = np.cross(delta1[0], delta1[1])
        self.assertTrue(_check_off_edge(dot1, cross1))

        point2 = [-1, -1]  # Off edge but on the line
        delta2 = edge - point2
        dot2 = delta2[0] @ delta2[1]
        cross2 = np.cross(delta2[0], delta2[1])
        self.assertTrue(_check_off_edge(dot2, cross2))

        point3 = [0.5, 0.5]  # Not off edge
        delta3 = edge - point3
        dot3 = delta3[0] @ delta3[1]
        cross3 = np.cross(delta3[0], delta3[1])
        self.assertFalse(_check_off_edge(dot3, cross3))

        point3 = [1, 1]  # Not off edge and on vertice
        delta3 = edge - point3
        dot3 = delta3[0] @ delta3[1]
        cross3 = np.cross(delta3[0], delta3[1])
        self.assertFalse(_check_off_edge(dot3, cross3))

        # Polygon case
        vertices = np.array([[0, 0], [0, 2], [3, 2], [3, 0], [2, 0], [2, 1], [1, 1], [1, 0]])

        point = [0.5, 0.5]         # Off edge
        delta_i = vertices - point
        delta_j = np.roll(delta_i, -2)
        dot = np.einsum('ij,ij->i', delta_i, delta_j)
        cross = np.cross(delta_j, delta_i)
        self.assertTrue(_check_off_edge(dot, cross))

        point = [1.5, 0]          # Off edge but on the line with one
        delta_i = vertices - point
        delta_j = np.roll(delta_i, -2)
        dot = np.einsum('ij,ij->i', delta_i, delta_j)
        cross = np.cross(delta_j, delta_i)
        self.assertTrue(_check_off_edge(dot, cross))

        point = [0.5, 0]          # On edge
        delta_i = vertices - point
        delta_j = np.roll(delta_i, -2)
        dot = np.einsum('ij,ij->i', delta_i, delta_j)
        cross = np.cross(delta_j, delta_i)
        self.assertFalse(_check_off_edge(dot, cross))

        point = [0, 0]          # On vertice
        delta_i = vertices - point
        delta_j = np.roll(delta_i, -2)
        dot = np.einsum('ij,ij->i', delta_i, delta_j)
        cross = np.cross(delta_j, delta_i)
        self.assertFalse(_check_off_edge(dot, cross))

    def test_wn_point_in_poly(self):
        self.assertTrue(_wn_point_in_poly(self.point1, self.vertices))
        self.assertFalse(_wn_point_in_poly(self.point2, self.vertices))
        self.assertFalse(_wn_point_in_poly(self.point3, self.vertices))
        self.assertTrue(_wn_point_in_poly(self.point4, self.vertices))

    def test_wn_vectorize(self):
        self.assertTrue(_wn_vectorize(self.point1, self.vertices))
        self.assertFalse(_wn_vectorize(self.point2, self.vertices))
        self.assertFalse(_wn_vectorize(self.point3, self.vertices))
        self.assertTrue(_wn_vectorize(self.point4, self.vertices))

    def test_wn_edge(self):
        self.assertTrue(_wn_edge(self.point1, self.vertices))
        self.assertFalse(_wn_edge(self.point2, self.vertices))
        self.assertFalse(_wn_edge(self.point3, self.vertices))
        self.assertTrue(_wn_edge(self.point4, self.vertices))

    def test_edge_case(self):
        self.assertFalse(_wn_edge(self.point5, self.vertices))
        self.assertFalse(_wn_edge(self.point6, self.vertices))

    def test_rc_point_in_poly(self):
        self.assertTrue(_rc_point_in_poly(self.point1, self.vertices))
        self.assertFalse(_rc_point_in_poly(self.point2, self.vertices))
        self.assertFalse(_rc_point_in_poly(self.point3, self.vertices))
        self.assertTrue(_rc_point_in_poly(self.point4, self.vertices))

    def test_rc_vectorize(self):
        self.assertTrue(_rc_vectorize(self.point1, self.vertices))
        self.assertFalse(_rc_vectorize(self.point2, self.vertices))
        self.assertFalse(_rc_vectorize(self.point3, self.vertices))
        self.assertTrue(_rc_vectorize(self.point4, self.vertices))

    def test_point_in_poly(self):
        # Convex U shape facing west
        vertices = np.array([[0, 0], [0, 2], [3, 2], [3, 0], [2, 0], [2, 1], [1, 1], [1, 0]])
        point1 = [1.5, 1.5]     # inside vertical area
        self.assertTrue(_point_in_poly(point1, vertices))
        self.assertTrue(_point_in_poly(point1, vertices, 'wn_edge'))
        self.assertTrue(_point_in_poly(point1, vertices, 'wn'))
        self.assertTrue(_point_in_poly(point1, vertices, 'rc'))
        self.assertTrue(_point_in_poly(point1, vertices, 'rc_vec'))
        self.assertTrue(_point_in_poly(point1, vertices, 'wn_vec'))

        point2 = [1.5, 0.5]     # outside between horizontal areas
        self.assertFalse(_point_in_poly(point2, vertices))
        self.assertFalse(_point_in_poly(point2, vertices, 'wn_edge'))
        self.assertFalse(_point_in_poly(point2, vertices, 'wn'))
        self.assertFalse(_point_in_poly(point2, vertices, 'rc'))
        self.assertFalse(_point_in_poly(point2, vertices, 'rc_vec'))
        self.assertFalse(_point_in_poly(point2, vertices, 'wn_vec'))

        point3 = [4, 2]         # outside above
        self.assertFalse(_point_in_poly(point3, vertices))
        self.assertFalse(_point_in_poly(point3, vertices, 'wn_edge'))
        self.assertFalse(_point_in_poly(point3, vertices, 'wn'))
        self.assertFalse(_point_in_poly(point3, vertices, 'rc'))
        self.assertFalse(_point_in_poly(point3, vertices, 'rc_vec'))
        self.assertFalse(_point_in_poly(point3, vertices, 'wn_vec'))

        point4 = [2.5, 0.5]     # inside top horizontal area
        self.assertTrue(_point_in_poly(point4, vertices))
        self.assertTrue(_point_in_poly(point4, vertices, 'wn_edge'))
        self.assertTrue(_point_in_poly(point4, vertices, 'wn'))
        self.assertTrue(_point_in_poly(point4, vertices, 'rc'))
        self.assertTrue(_point_in_poly(point4, vertices, 'rc_vec'))
        self.assertTrue(_point_in_poly(point4, vertices, 'wn_vec'))

    def test_import_file(self):
        # TODO: mock the input files
        name, vertices = _import_file("./data/SICS.txt")
        self.assertEqual("site", name)
        self.assertTrue(type(vertices) == list)
        self.assertTrue(len(vertices) > 0)

        name1, vertices1 = _import_file("./data/sics.json")
        self.assertTrue(type(name1) == str)
        self.assertTrue(name1 == 'sics')
        self.assertTrue(type(vertices1) == list)
        self.assertTrue(len(vertices1) > 0)

        self.assertRaises(FileNotFoundError, lambda: _import_file("/path-not-exist"))
        self.assertRaises(ValueError, lambda: _import_file('./data/bad.txt'))
        self.assertRaises(ValueError, lambda: _import_file('./data/map.png'))
