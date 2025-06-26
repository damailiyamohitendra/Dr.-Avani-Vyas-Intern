import math
from collections import Counter


def calculate_entropy(data, target_index):
    if not data:
        return 0.0

    labels = [row[target_index] for row in data]
    label_counts = Counter(labels)

    total_samples = len(labels)
    entropy = 0.0
    for count in label_counts.values():
        probability = count / total_samples
        if probability > 0:
            entropy -= probability * math.log2(probability)
    return entropy


def split_data(dataset, feature_index, feature_value):
    data_left = [row for row in dataset if row[feature_index] == feature_value]
    data_right = [row for row in dataset if row[feature_index] != feature_value]
    return data_left, data_right


def most_common_label(data, target_index):
    if not data:
        return None

    labels = [row[target_index] for row in data]
    return Counter(labels).most_common(1)[0][0]


def get_best_split(dataset, target_index):
    initial_entropy = calculate_entropy(dataset, target_index)
    best_information_gain = 0.0
    best_feature_index = -1
    best_feature_value = None
    num_columns = len(dataset[0])
    for feature_idx in range(num_columns):
        if feature_idx == target_index:
            continue
        unique_values = set(row[feature_idx] for row in dataset)

        for value in unique_values:
            left_data, right_data = split_data(dataset, feature_idx, value)
            if not left_data or not right_data:
                continue

            prob_left = len(left_data) / len(dataset)
            prob_right = len(right_data) / len(dataset)

            entropy_left = calculate_entropy(left_data, target_index)
            entropy_right = calculate_entropy(right_data, target_index)

            weighted_avg_entropy = (prob_left * entropy_left) + (prob_right * entropy_right)
            information_gain = initial_entropy - weighted_avg_entropy

            if information_gain > best_information_gain:
                best_information_gain = information_gain
                best_feature_index = feature_idx
                best_feature_value = value

    return best_feature_index, best_feature_value, best_information_gain


def build_tree(dataset, target_index):
    if not dataset:
        return None

    current_labels = [row[target_index] for row in dataset]

    if len(set(current_labels)) == 1:
        return (None, None, None, None, current_labels[0])

    best_feature_index, best_feature_value, best_information_gain = \
        get_best_split(dataset, target_index)

    if best_information_gain == 0:
        return (None, None, None, None, most_common_label(dataset, target_index))

    true_data, false_data = split_data(dataset, best_feature_index, best_feature_value)

    true_branch = build_tree(true_data, target_index)
    false_branch = build_tree(false_data, target_index)

    return (best_feature_index, best_feature_value, true_branch, false_branch, None)


def classify(observation, tree_node_tuple):
    feature_idx, feature_val, true_b, false_b, prediction_val = tree_node_tuple

    if prediction_val is not None:
        return prediction_val

    value_in_observation = observation[feature_idx]

    if value_in_observation == feature_val:
        return classify(observation, true_b)
    else:
        return classify(observation, false_b)


def calculate_accuracy(true_labels, predicted_labels):
    """
    Calculates the accuracy score.
    """
    if not true_labels:
        return 0.0

    correct_predictions = 0
    for i in range(len(true_labels)):
        if true_labels[i] == predicted_labels[i]:
            correct_predictions += 1
    return correct_predictions / len(true_labels)


def print_tree(tree_node_tuple, feature_names, indent=""):
    if tree_node_tuple is None:
        print(f"{indent}--> (Empty Branch)")
        return

    feature_idx, feature_val, true_b, false_b, prediction_val = tree_node_tuple

    if prediction_val is not None:
        print(f"{indent}Leaf (Predicts): '{prediction_val}'")
        return

    print(f"{indent}IF {feature_names[feature_idx]} == '{feature_val}':")
    print(f"{indent}  --> TRUE branch:")
    print_tree(true_b, feature_names, indent + "    ")
    print(f"{indent}  --> FALSE branch:")
    print_tree(false_b, feature_names, indent + "    ")

if __name__ == "__main__":
    data = [
        ['Sunny', 'Hot', 'High', 'Weak', 'No'],
        ['Sunny', 'Hot', 'High', 'Strong', 'No'],
        ['Overcast', 'Hot', 'High', 'Weak', 'Yes'],
        ['Rain', 'Mild', 'High', 'Weak', 'Yes'],
        ['Rain', 'Cool', 'Normal', 'Weak', 'Yes'],
        ['Rain', 'Cool', 'Normal', 'Strong', 'No'],
        ['Overcast', 'Cool', 'Normal', 'Strong', 'Yes'],
        ['Sunny', 'Mild', 'High', 'Weak', 'No'],
        ['Sunny', 'Cool', 'Normal', 'Weak', 'Yes'],
        ['Rain', 'Mild', 'Normal', 'Weak', 'Yes'],
        ['Sunny', 'Mild', 'Normal', 'Strong', 'Yes'],
        ['Overcast', 'Mild', 'High', 'Strong', 'Yes'],
        ['Overcast', 'Hot', 'Normal', 'Weak', 'Yes'],
        ['Rain', 'Mild', 'High', 'Strong', 'No']
    ]

    feature_names = ['Outlook', 'Temperature', 'Humidity', 'Wind']
    target_index = len(data[0]) - 1

    print("Step 1: Building the Decision Tree...")

    my_decision_tree_tuple = build_tree(data, target_index)
    print("Tree Building Complete!")

    print("\n--- Step 2: Visualizing the Decision Tree Structure ---")

    print_tree(my_decision_tree_tuple, feature_names)

    print("\n--- Step 3: Making Predictions on New Data ---")

    test_observations = [
        ['Sunny', 'Cool', 'High', 'Strong', 'No'],  # Expected: 'No'
        ['Rain', 'Mild', 'Normal', 'Weak', 'Yes'],  # Expected: 'Yes'
        ['Overcast', 'Hot', 'High', 'Weak', 'Yes'],  # Expected: 'Yes'
        ['Rain', 'Hot', 'High', 'Strong', 'No']      # Expected: 'No'
    ]

    for i, obs in enumerate(test_observations):
        features_only = obs[:-1]
        actual_label = obs[-1]

        predicted_label = classify(features_only, my_decision_tree_tuple)
        print(f"Observation {i + 1}: {features_only}")
        print(f"  -> Predicted: {predicted_label}, Actual: {actual_label}")

    print("\n--- Step 4: Calculating Simple Training Accuracy ---")

    true_labels_for_accuracy = [row[target_index] for row in data]
    features_only_for_prediction = [row[:-1] for row in data]

    predictions_on_training_data = []
    for obs in features_only_for_prediction:
        predictions_on_training_data.append(classify(obs, my_decision_tree_tuple))

    accuracy = calculate_accuracy(true_labels_for_accuracy, predictions_on_training_data)
    print(f"Accuracy on Training Data: {accuracy:.2f}")
    