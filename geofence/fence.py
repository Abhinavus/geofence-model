import sys
import os
sys.path.append(os.path.dirname(os.path.abspath('geofence\fence.py')))

from geofence.utils import _point_in_poly, _import_file
import numpy as np


class Fence(object):
    """
        The Fence object contains all the vertices information
            and method for detecting whether a point is inside of the geo-fence

        Example:
            >>> point = [59.405014, 17.949540]
            >>> fence = Fence()
            >>> fence.set_vertices_from_file('data/SICS.txt')
            >>> fence.detect(point)
            True
    """

    def __init__(self, vs=None, site=""):
        self._site = site
        self._vertices = []
        if vs:
            self.vertices = vs

    @property
    def site(self):
        return self._site

    @site.setter
    def site(self, site: str):
        self._site = site

    @property
    def vertices(self):
        return self._vertices

    @vertices.setter
    def vertices(self, vs):
        assert len(vs) >= 3, "polygon need to have at least three vertices"
        self._vertices = np.array(vs)

    def set_vertices_from_file(self, file):
        self._site, self.vertices = _import_file(file)

    def detect(self, point: list, algo=None):
        assert len(point) == 2, "coordinates must be latitude, longitude"
        return _point_in_poly(point, self.vertices, algo)
