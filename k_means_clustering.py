import random
import math


def euclidean_distance(point1, point2):
    distance_squared = 0
    for i in range(len(point1)):
        distance_squared += (point1[i] - point2[i]) ** 2
    return math.sqrt(distance_squared)


def initialize_centroids(data_points, k):
    if k > len(data_points):
        raise ValueError("K (number of clusters) cannot be greater than the total number of data points.")

    random_indices = random.sample(range(len(data_points)), k)

    centroids = {}
    for i, data_index in enumerate(random_indices):
        centroids[i] = data_points[data_index]
    return centroids


def assign_to_clusters(data_points, current_centroids):
    assignments = []
    for point in data_points:
        distances_to_centroids = [
            euclidean_distance(point, current_centroids[centroid_id])
            for centroid_id in current_centroids
        ]

        min_distance = min(distances_to_centroids)
        closest_centroid_id = distances_to_centroids.index(min_distance)

        assignments.append(closest_centroid_id)
    return assignments


def update_centroids(data_points, cluster_assignments, k, old_centroids):
    new_centroids = {}

    for centroid_id in range(k):
        points_in_cluster = [
            data_points[i] for i in range(len(data_points))
            if cluster_assignments[i] == centroid_id
        ]
        if not points_in_cluster:
            new_centroids[centroid_id] = old_centroids[centroid_id]
            continue
        num_dimensions = len(points_in_cluster[0])
        sum_coords = [0] * num_dimensions

        for point in points_in_cluster:
            for d in range(num_dimensions):
                sum_coords[d] += point[d]

        new_centroid_coords = [
            coord_sum / len(points_in_cluster)
            for coord_sum in sum_coords
        ]

        new_centroids[centroid_id] = new_centroid_coords
    return new_centroids


def run_kmeans(data_points, k=3, max_iter=100, tolerance=1e-4):
    if not data_points:
        print("Warning: No data provided to K-Means. Returning empty results.")
        return {}, []
    centroids = initialize_centroids(data_points, k)
    cluster_assignments = []
    for iteration_num in range(max_iter):
        centroids_start_iteration = dict(centroids)
        cluster_assignments = assign_to_clusters(data_points, centroids)
        centroids = update_centroids(data_points, cluster_assignments, k, centroids_start_iteration)

        has_converged = True
        for centroid_id in centroids:
            old_c = centroids_start_iteration[centroid_id]
            new_c = centroids[centroid_id]
            distance_this_centroid_moved = euclidean_distance(old_c, new_c)
            if distance_this_centroid_moved > tolerance:
                has_converged = False
                break

        if has_converged:
            print(f"K-Means converged (stopped moving) after {iteration_num + 1} iterations.")
            break
        else:
            print(f"K-Means reached the maximum number of iterations ({max_iter}) without fully converging.")
    return centroids, cluster_assignments


if __name__ == "__main__":
    data = [
        [1.0, 1.0], [1.5, 2.0], [3.0, 4.0], [5.0, 7.0],
        [3.5, 5.0], [4.5, 5.0], [3.5, 4.5], [6.0, 8.0],
        [9.0, 9.0], [8.0, 8.5], [9.5, 7.0], [7.5, 6.5]
    ]

    print("--- K-Means Clustering ---")
    print("The data points used here are:")
    for point in data:
        print(f"   {point}")

    num_clusters_to_find = 3
    maximum_iterations = 100
    movement_tolerance = 1e-4

    print(f"\nStarting K-Means with K = {num_clusters_to_find}...")
    final_trained_centroids, final_cluster_assignments = run_kmeans(
        data,
        k=num_clusters_to_find,
        max_iter=maximum_iterations,
        tolerance=movement_tolerance
    )

    print("\n--- Clustering Results ---")
    print("The final centroids for each cluster are:")
    for cluster_id, centroid_coords in final_trained_centroids.items():
        print(f"   Cluster {cluster_id}: {[round(c, 2) for c in centroid_coords]}")

    print("\nAnd here's which cluster each of our original data points was assigned to:")
    for i, assignment in enumerate(final_cluster_assignments):
        print(f"   Point {data[i]} was assigned to Cluster {assignment}")