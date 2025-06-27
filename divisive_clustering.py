import random
import math


def euclidean_distance(point1, point2):
    if len(point1) != len(point2):
        raise ValueError("Points must have the same number of dimensions for distance calculation.")
    distance_squared = 0
    for i in range(len(point1)):
        distance_squared += (point1[i] - point2[i]) ** 2
    return math.sqrt(distance_squared)


def calculate_average_dissimilarity(cluster_points):
    n = len(cluster_points)
    if n <= 1:
        return 0.0

    total_dissimilarity = 0.0
    num_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            total_dissimilarity += euclidean_distance(cluster_points[i], cluster_points[j])
            num_pairs += 1

    return total_dissimilarity / num_pairs if num_pairs > 0 else 0.0


def _split_cluster_by_farthest_points(cluster_to_split):
    n = len(cluster_to_split)
    if n < 2:
        return cluster_to_split, []

    max_dist = -1.0
    farthest_point1 = None
    farthest_point2 = None

    for i in range(n):
        for j in range(i + 1, n):
            dist = euclidean_distance(cluster_to_split[i], cluster_to_split[j])
            if dist > max_dist:
                max_dist = dist
                farthest_point1 = cluster_to_split[i]
                farthest_point2 = cluster_to_split[j]

    if farthest_point1 is None or farthest_point2 is None:
        return cluster_to_split, []
    sub_cluster1 = [farthest_point1]
    sub_cluster2 = [farthest_point2]

    for point in cluster_to_split:
        if point == farthest_point1 or point == farthest_point2:
            continue
        dist_to_p1 = euclidean_distance(point, farthest_point1)
        dist_to_p2 = euclidean_distance(point, farthest_point2)

        if dist_to_p1 <= dist_to_p2:
            sub_cluster1.append(point)
        else:
            sub_cluster2.append(point)

    return sub_cluster1, sub_cluster2


def run_divisive_clustering(data_points, k):
    if not data_points:
        print("NO DATA PROVIDED")
        return []

    if k < 1:
        raise ValueError("Number of clusters must be at least 1")
    if k > len(data_points):
        print(
            f"Number of cluster ({k}) are GREATER than data points ({len(data_points)}). will return {len(data_points)} each point as one cluster.")
        return [[p] for p in data_points]

    current_clusters = [data_points]
    print(f"Initial state: One big cluster with {len(data_points)} points.")

    while len(current_clusters) < k:
        best_cluster_idx_to_split = -1
        max_avg_dissimilarity = -1.0

        for i, cluster in enumerate(current_clusters):
            if len(cluster) <= 1:
                continue
            avg_dissimilarity = calculate_average_dissimilarity(cluster)

            if avg_dissimilarity > max_avg_dissimilarity:
                max_avg_dissimilarity = avg_dissimilarity
                best_cluster_idx_to_split = i

        if best_cluster_idx_to_split == -1:
            print(
                f"Cannot split further. Reached {len(current_clusters)} clusters (all are singletons or un-splittable).")
            break

        cluster_to_divide = current_clusters[best_cluster_idx_to_split]

        new_sub_cluster1, new_sub_cluster2 = _split_cluster_by_farthest_points(cluster_to_divide)

        if not new_sub_cluster1 or not new_sub_cluster2:
            print(
                f"Skipping split of cluster {best_cluster_idx_to_split} as it resulted in an ineffective split. Halting divisive clustering.")
            break

        temp_clusters = []
        for i, cluster in enumerate(current_clusters):
            if i == best_cluster_idx_to_split:
                temp_clusters.append(new_sub_cluster1)
                temp_clusters.append(new_sub_cluster2)
            else:
                temp_clusters.append(cluster)

        current_clusters = temp_clusters
        print(f"Split a cluster. Current number of clusters: {len(current_clusters)}")

    print(f"\nDivisive Clustering finished. Reached {len(current_clusters)} clusters.")
    return current_clusters


def print_final_clusters(clusters):
    print("\n--- Final Clusters ---")
    if not clusters:
        print("No clusters to display.")
        return

    for i, cluster in enumerate(clusters):
        print(f"Cluster {i + 1}: (Contains {len(cluster)} points)")
        for point in cluster:
            print(f"  {point}")


if __name__ == "__main__":
    data = [
        [1.0, 1.0], [1.5, 2.0],
        [3.0, 4.0], [3.5, 5.0], [4.5, 5.0], [3.5, 4.5],
        [5.0, 7.0], [6.0, 8.0],
        [7.5, 6.5], [8.0, 8.5], [9.0, 9.0], [9.5, 7.0]
    ]

    print(" Divisive Hierarchical Clustering!")
    print("Original Data Points:")
    for point in data:
        print(f"   {point}")

    num_clusters = 4

    print(f"\nStarting Divisive Clustering to get {num_clusters} clusters...")
    final_clusters = run_divisive_clustering(data, k=num_clusters)
    print_final_clusters(final_clusters)

    print(" Using K with all points->")
    final_clusters_k_all = run_divisive_clustering(data, k=len(data))
    print_final_clusters(final_clusters_k_all)
    