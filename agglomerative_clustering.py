import math


def distance(point1, point2):
    if len(point1) != len(point2):
        raise ValueError("Points don't have same number of dimensions")

    dist = 0
    for i in range(len(point1)):
        dist += abs(point1[i] - point2[i])
    return dist


def calc_cluster_distance(c1, c2):
    min_distance_cluster = math.inf

    for p1 in c1:
        for p2 in c2:
            point_distance = distance(p1, p2)

            if point_distance < min_distance_cluster:
                min_distance_cluster = point_distance
    return min_distance_cluster


def initialize_clusters(data_points):
    initial_clusters = []

    for point in data_points:
        initial_clusters.append([point])
    return initial_clusters


def find_closest_cluster_pair(current_clusters):
    min_dist = math.inf
    pair_index = None

    num_clusters = len(current_clusters)

    for i in range(num_clusters - 1):
        c_i = current_clusters[i]
        for j in range(i + 1, num_clusters):
            c_j = current_clusters[j]

            dist = calc_cluster_distance(c_i, c_j)
            if dist < min_dist:
                min_dist = dist
                pair_index = (i, j)
    return pair_index


def merge_clusters(current_clusters, index1, index2):
    merged_cluster = current_clusters[index1] + current_clusters[index2]

    new_list_clusters = []
    new_list_clusters.append(merged_cluster)

    for i, cluster in enumerate(current_clusters):
        if i != index1 and i != index2:
            new_list_clusters.append(cluster)
    return new_list_clusters


def run_agglomerative_clustering(data_points, k):
    if not data_points:
        print("No data provided!")
        return []

    current_clusters = initialize_clusters(data_points)
    print(f"Initial Clusters where each data point is a cluster: {current_clusters}")
    while len(current_clusters) > k:
        id1, id2 = find_closest_cluster_pair(current_clusters)

        if id1 is None:
            print("No more clusters to merge!")
            break

        current_clusters = merge_clusters(current_clusters, id1, id2)
        print(f"\n Clustering finished! Reached {len(current_clusters)} clusters.")
        return current_clusters


def print_final_clusters(clusters):
    print("\n FINAL CLUSTERS ")
    if not clusters:
        print("No clusters available")
        return
    for i, cluster in enumerate(clusters):
        print(f"\n Cluster {i + 1}")
        for point in cluster:
            print(f" {point}")


if __name__ == "__main__":
    data = [
        [1.0, 1.0], [1.5, 2.0], [3.0, 4.0], [5.0, 7.0],
        [3.5, 5.0], [4.5, 5.0], [3.5, 4.5], [6.0, 8.0],
        [9.0, 9.0], [8.0, 8.5], [9.5, 7.0], [7.5, 6.5]
    ]

    print("--- Agglomerative Hierarchical Clustering (Manhattan Distance, Single Linkage) ---")
    print("Original Data Points:")
    for point in data:
        print(f" {point}")

    num_clusters = 3
    final_clusters = run_agglomerative_clustering(data, k=num_clusters)
    print_final_clusters(final_clusters)