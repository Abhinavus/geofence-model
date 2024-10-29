import json
import numpy as np
from pathlib import Path


def _point_in_poly(point: list, vertices: list, algo="wc_edge"):
    """ determine if a point is in the polygon

    Args:
        point -- latitude, longitude of the point to test
        vertices -- defined region [[lat0, lon0], [lat1, lon1], ...], counter-clock order
        algo -- "rc"-ray_casting
                "wn"-winding_number
                "rc_vec"-vectorized rc
                "wn_vec"-vectorized wn
                "wn_edge"-default, capable of edge cases

    Returns:
        True if the point is inside of the polygon, boundary points undefined
    """
    if algo == 'rc':
        return _rc_point_in_poly(point, vertices)
    if algo == 'wc':
        return _wn_point_in_poly(point, vertices)
    if algo == 'rc_vec':
        return _rc_vectorize(point, vertices)
    if algo == 'wn_vec':
        return _wn_vectorize(point, vertices)
    else:
        return _wn_edge(point, vertices)


def _wn_point_in_poly(point: list, vertices: list):
    """ determine if a point is in the polygon using winding number algorithm

    Returns:
        True if the point is inside of the polygon (when wn != 0). Undefined on boundary
    """
    wn = 0
    n = len(vertices)

    for i in range(n):
        j = (i + 1) % n
        if vertices[i][0] <= point[0]:
            if (vertices[j][0] > point[0]) and (
                    _is_left(*point, *vertices[i], *vertices[j]) > 0):
                wn += 1
        else:
            if (vertices[j][0] <= point[0]) and (
                    _is_left(*point, *vertices[i], *vertices[j]) < 0):
                wn -= 1
    return wn != 0


def _wn_vectorize(point: list, vertices: list):
    wn = 0
    dx = vertices[:, 0] - point[0]
    dy = vertices[:, 1] - point[1]

    dy_n, dx_n = np.roll(dy, -1), np.roll(dx, -1)
    is_left = dx * dy_n - dx_n * dy
    wn += np.sum((dy <= 0) * (dy_n > 0) * np.sign(is_left))
    wn += np.sum((dy > 0) * (dy_n <= 0) * np.sign(is_left))

    return wn != 0


def _wn_edge(point:list, vertices: np.ndarray):
    """ Determines if a point is in the polygon using
        a modified winding number algorithm which returns
        False if the point is on the boundary.

    Returns: True if the point is inside of the polygon (when wn != 0).
             Edge cases return False.
    """

    wn = 0
    delta_i = vertices - point     # point-to-edge vectors, columns corresponding to dy, dx
    delta_j = np.roll(delta_i, -2)
    dot = np.einsum('ij,ij->i', delta_i, delta_j)
    cross = np.cross(delta_i, delta_j)

    dyi, dyj = delta_i[:, 0], delta_j[:, 0]
    valid_up = 1 * (dyi <= 0) * (0 < dyj) * (cross > 0)
    valid_down = 1 * (0 < dyi) * (dyj <= 0) * (cross < 0)

    wn = np.sum(valid_up - valid_down) * _check_off_edge(dot, cross)

    return wn != 0


def _rc_point_in_poly(point: list, vertices: list):
    """ determine if a point is in the polygon using ray casting algorithm

    Returns:
      True if the point is inside of the polygon. Undefined on the boundary
    """
    inside = False
    n = len(vertices)
    j = n - 1
    lat, lon = point

    for i in range(n):
        if ((vertices[i][0] < lat <= vertices[j][0]
             or vertices[j][0] < lat <= vertices[i][0])
                and (vertices[i][1] <= lon or vertices[j][1] <= lon)):
            k = (lat - vertices[i][0]) / (vertices[j][0] - vertices[i][0])
            inside ^= (vertices[i][1] + k * (vertices[j][1] - vertices[i][1]) < lon)
        j = i

    return inside


def _rc_vectorize(point: list, vertices: list):
    inside = False
    n = len(vertices)

    dxi = vertices[:, 0] - point[0]
    dyi = vertices[:, 1] - point[1]

    dyj, dxj = np.roll(dyi, -1), np.roll(dxi, -1)

    xi, yi = vertices[:, 0], vertices[:, 1]
    xj, yj = np.roll(xi, -1), np.roll(yi, -1)

    indicators = (dyi * dyj > 0) * (
            point[0] < (xi + (point[1] - yi) * np.true_divide(
        (xj - xi), (yj - yi), where=((yj - yi)) != 0)))

    inside ^= np.logical_xor.reduce(indicators)
    return inside


def _is_left(y0, x0, y1, x1, y2, x2):
    """ determine if a point on the left of a line using cross-product rule

    Args:
        y0, x0 -- latitude and longitude of point to test
        y1, x1 -- latitude and longitude of end point 1 of line
        y2, x2 -- latitude and longitude of end point 2 of line

    Returns:
        >0 if it is on the left
        <0 if it is on the right
        =0 if it is on the line
    """
    return (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)


def _check_off_edge(dot, cross):
    """ check if point P is off the edge bounded by V_i and V_j
        using the dot product and cross product of PV_i and PV_j

    Args:
        dot: <PV_i, PV_j>
        cross: PV_i x PV_j

    Returns: True if the point is NOT on the edge

    """
    return np.sum((dot <= 0) * (cross == 0)) == 0


def _import_file(file_path: str):
    """ import coordinates from file

    Args:
      file_path -- file path
        example .txt file:
            59.4048184072506,17.9478923473923
            59.4043815528131,17.9485360775559
            59.404422508156,17.9486433659165
        example .json file can be created at http://geo.jasparke.net/
            [
                {
                    "name": "sics",
                    "color": "#6CB1E1",
                    "id": 0,
                    "path": [
                        [
                            59.4048182070281,
                            17.9478945561005
                        ],
                        [
                            59.404377257051,
                            17.9485382862641
                        ],
                        [
                            59.4046025108524,
                            17.9491310544564
                        ],
                        [
                            59.4046598479445,
                            17.949050588186
                        ],
                    ]
                }
            ]

    Returns:
        name of the site from .json files or "site" as default for .txt files
        geo coordinates of the vertices [[lat0, lon0], [lat1, lon1], ...]
    """
    try:
        with open(file_path) as file:
            if Path(file_path).suffix == '.json':
                data = json.load(file)[0]
                return data.get('name'), data.get('path')
            elif Path(file_path).suffix == '.txt':
                lines = file.readlines()
                vertices = []
                for line in lines:
                    vertices.append([float(x) for x in line.strip().split(',')])
                return "site", vertices
            else:
                raise ValueError("File type not supported")
    except FileNotFoundError:
        raise
    except ValueError:
        raise
