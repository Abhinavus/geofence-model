import sys
import os
import json
import math
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath('geofence\fence.py')))

from geofence.fence import Fence


def generate_polygon_vertices(center_point, radius, num_vertices=8):
    R = 6371000  # Radius of the Earth in meters
    lat_center, lon_center = math.radians(center_point[0]), math.radians(center_point[1])

    vertices = []
    angle_increment = 2 * math.pi / num_vertices  # Angle between each vertex in radians

    for i in range(num_vertices):
        angle = i * angle_increment

        lat_vertex = math.asin(math.sin(lat_center) * math.cos(radius / R) + 
                               math.cos(lat_center) * math.sin(radius / R) * math.cos(angle))
        lon_vertex = lon_center + math.atan2(math.sin(angle) * math.sin(radius / R) * math.cos(lat_center),
                                             math.cos(radius / R) - math.sin(lat_center) * math.sin(lat_vertex))
        
        # Convert radians back to degrees
        lat_vertex = math.degrees(lat_vertex)
        lon_vertex = math.degrees(lon_vertex)
        
        vertices.append((lat_vertex, lon_vertex))

    return vertices


def create_geofences_from_json(json_file, radius, num_vertices=8):
    try:
        with open(json_file, 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        print(f"Error: File {json_file} not found.")
        return []
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from file {json_file}.")
        return []

    geofences = []

    for item in data:
        path = item.get('path', [])
        
        for location in path:
            lat, lon = location[0], location[1]  # Adjust based on JSON structure
            vertices = generate_polygon_vertices((lat, lon), radius, num_vertices)
            geofences.append(vertices)

    return geofences


def add_geo_noise(point, epsilon, radius):
    """
    Adds Laplace noise to a geographic point for geo-indistinguishability.
    
    Args:
        point (list): [latitude, longitude] of the point.
        epsilon (float): Privacy budget parameter, controls the amount of noise.
        radius (float): The radius around the point in which noise will be applied.
        
    Returns:
        list: A new point with noise applied, in the form [latitude, longitude].
    """
    lat, lon = point
    
    # Sample noise for angle and distance
    theta = np.random.uniform(0, 2 * np.pi)  # Random angle
    d = np.random.laplace(scale=radius / (epsilon * np.sqrt(2)))   # Exponential distribution for geo-indistinguishability
    
    # Convert from meters to degrees (approximately)
    earth_radius = 6371000  # meters
    delta_lat = d / earth_radius * (180 / np.pi)
    delta_lon = d / (earth_radius * np.cos(np.pi * lat / 180)) * (180 / np.pi)
    
    # Apply noise to the original point
    noisy_lat = lat + delta_lat * np.sin(theta)
    noisy_lon = lon + delta_lon * np.cos(theta)
    
    return [noisy_lat, noisy_lon]


# Define file path and the point
file_path = 'C:\\Project\\geofencing\\data\\sics.json'  # JSON file containing lat/long points of geofences
point = [50.3744352442476256, -4.13210744929211]  # The point to test

# Privacy parameters for geo-indistinguishability
epsilon = 10 # Privacy budget parameter (lower means more noise, higher means less noise)
radius = 200   # Radius in meters within which the noise is applied

# Create Fence object
fence = Fence()

# Load vertices (geofences) from the JSON file and set them in the Fence object
geofences = create_geofences_from_json(file_path, radius=1000, num_vertices=8)

# Apply geo-indistinguishable noise to the point
noisy_point = add_geo_noise(point, epsilon, radius)
print(f"Original point: {point}")
print(f"Noisy point (geo-indistinguishable): {noisy_point}")

# Loop through each geofence and detect if the noisy point is inside any polygon
point_inside_any_fence = False

# Loop through each geofence and apply different detection algorithms
for i, vertices in enumerate(geofences):
    fence.vertices = vertices

    # Run detection using multiple algorithms and collect results
    result_rc = fence.detect(noisy_point, algo='rc')           # Ray-Casting
    result_wc = fence.detect(noisy_point, algo='wc')           # Winding number
    result_rc_vec = fence.detect(noisy_point, algo='rc_vec')   # Vectorized Ray-Casting
    result_wn_vec = fence.detect(noisy_point, algo='wn_vec')   # Vectorized Winding number

    # Voting system: Majority of algorithms must agree that the point is inside the geofence
    vote_results = [result_rc, result_wc, result_rc_vec, result_wn_vec]
    votes_in_fence = sum(vote_results)

    if votes_in_fence >= 3:  # If 3 or more algorithms detect the point inside
        point_inside_any_fence = True
        print(f"\nResult: The noisy point {noisy_point} is INSIDE Geofence {i+1}.")
        break  # Exit the loop if the point is inside at least one geofence

if not point_inside_any_fence:
    print(f"\nResult: The noisy point {noisy_point} is OUTSIDE all geofences.")
