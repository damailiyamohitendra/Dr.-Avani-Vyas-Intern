import math

training_data = [
    [1.0, 2.0, 'A'],
    [2.0, 3.0, 'A'],
    [3.0, 1.0, 'A'],
    [4.0, 4.0, 'B'],
    [5.0, 3.0, 'B'],
    [6.0, 5.0, 'B'],
    [1.5, 2.5, 'A'],
    [4.5, 3.5, 'B']
]

print("--- Training Data ---")
for row in training_data:
    print(f"Features: {row[:-1]}, Label: {row[-1]}")
print("-" * 30)


def euclidean_distance(point1, point2):
    distance = 0
    for i in range(len(point1)):
        distance += (point1[i] - point2[i]) ** 2
    return math.sqrt(distance)


def get_k_nearest_neighbors(new_point, data, k):
    for data_row in data:
        distances=[]
        features_of_data_row = data_row[:-1]
        dist = euclidean_distance(new_point, features_of_data_row)
        distances.append((dist, data_row))

    distances.sort(key=lambda x: x[0])

    k_nearest = []

    for i in range(min(k, len(distances))):
        k_nearest.append(distances[i][1])
    return k_nearest


def predict_knn(new_point, data, k):
    if not data:
        return "No training data available."

    neighbors = get_k_nearest_neighbors(new_point, data, k)

    class_votes = {}
    for neighbor_row in neighbors:
        label = neighbor_row[-1]
        class_votes[label] = class_votes.get(label, 0) + 1

    predicted_class = None
    max_votes = -1

    for label, votes in class_votes.items():
        if votes > max_votes:
            max_votes = votes
            predicted_class = label

    return predicted_class, neighbors

test_point = [2.5, 2.0]
k_value = 3

print(f"\n--- Classifying New Point: {test_point} with K={k_value} ---")

predicted_class, neighbors_found = predict_knn(test_point, training_data, k_value)

print(f"\nK-Nearest Neighbors Found ({k_value} neighbors):")
for neighbor in neighbors_found:
    print(f"  - Features: {neighbor[:-1]}, Label: {neighbor[-1]}")

print(f"\nPredicted Class for {test_point}: {predicted_class}")
print("-" * 30)

test_point_2 = [5.0, 4.0]
k_value_2 = 5

print(f"\n--- Classifying New Point: {test_point_2} with K={k_value_2} ---")
predicted_class_2, neighbors_found_2 = predict_knn(test_point_2, training_data, k_value_2)

print(f"\nK-Nearest Neighbors Found ({k_value_2} neighbors):")
for neighbor in neighbors_found_2:
    print(f"  - Features: {neighbor[:-1]}, Label: {neighbor[-1]}")

print(f"\nPredicted Class for {test_point_2}: {predicted_class_2}")
print("-" * 30)
