from collections import Counter


def calculate_gini_impurity(data, target_index):
    if not data:
        return 0.0
    labels = [row[target_index] for row in data]
    label_counts = Counter(labels)

    gini = 1.0
    total_samples = len(labels)

    for count in label_counts.values():
        probability = count / total_samples
        gini -= probability ** 2
    return gini

def split_data(data, feature_index, threshold):
    left_subset = []  # feature_value <= threshold
    right_subset = [] # feature_value > threshold

    for row in data:
        if row[feature_index] <= threshold:
            left_subset.append(row)
        else:
            right_subset.append(row)
    return left_subset, right_subset


def get_most_common_label(data, target_index):
    if not data:
        return None

    labels = [row[target_index] for row in data]
    return Counter(labels).most_common(1)[0][0]


def find_best_split(data, target_index):
    num_features = len(data[0]) - 1
    current_gini = calculate_gini_impurity(data, target_index)
    best_gini_gain = -1.0
    best_split_info = None

    for feature_index in range(num_features):
        unique_values = sorted(list(set([row[feature_index] for row in data])))

        thresholds = []
        if len(unique_values) > 1:
            for i in range(len(unique_values) - 1):
                thresholds.append((unique_values[i] + unique_values[i + 1]) / 2)
        else:
            continue

        for threshold in thresholds:
            left_subset, right_subset = split_data(data, feature_index, threshold)

            if not left_subset or not right_subset:
                continue

            total_samples = float(len(data))
            gini_left = calculate_gini_impurity(left_subset, target_index)
            gini_right = calculate_gini_impurity(right_subset, target_index)

            weighted_avg_gini = (len(left_subset) / total_samples * gini_left) + \
                                (len(right_subset) / total_samples * gini_right)

            gini_gain = current_gini - weighted_avg_gini

            if gini_gain > best_gini_gain:
                best_gini_gain = gini_gain
                best_split_info = (feature_index, threshold, left_subset, right_subset)

    return best_gini_gain, best_split_info


def build_tree(data, target_index, max_depth=None, min_samples_split=2, current_depth=0):
    if not data:
        return None

    current_labels = [row[target_index] for row in data]
    if len(set(current_labels)) == 1:
        return (None, None, None, None, current_labels[0])

    if max_depth is not None and current_depth >= max_depth:
        return (None, None, None, None, get_most_common_label(data, target_index))

    if len(data) < min_samples_split:
        return (None, None, None, None, get_most_common_label(data, target_index))

    best_gini_gain, split_info = find_best_split(data, target_index)

    if split_info is None or best_gini_gain <= 0:
        return (None, None, None, None, get_most_common_label(data, target_index))

    feature_index, threshold, left_data, right_data = split_info

    left_child = build_tree(left_data, target_index, max_depth, min_samples_split, current_depth + 1)
    right_child = build_tree(right_data, target_index, max_depth, min_samples_split, current_depth + 1)

    return (feature_index, threshold, left_child, right_child, None)


def predict_single(node_tuple, sample):
    feature_idx, threshold_val, left_branch, right_branch, predicted_val = node_tuple

    if predicted_val is not None:
        return predicted_val

    sample_feature_value = sample[feature_idx]

    if sample_feature_value <= threshold_val:
        return predict_single(left_branch, sample)
    else:
        return predict_single(right_branch, sample)

def predict(tree_root_tuple, data_to_predict):
    predictions = []
    for sample in data_to_predict:
        predictions.append(predict_single(tree_root_tuple, sample))
    return predictions


def accuracy_score(y_true, y_pred):
    if not y_true:
        return 0.0

    correct = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            correct += 1
    return correct / len(y_true)


def print_tree(node_tuple, indent=""):
    if node_tuple is None:
        print(f"{indent}--> (Empty Branch)")
        return

    feature_idx, threshold_val, left_branch, right_branch, predicted_val = node_tuple

    if predicted_val is not None:
        print(f"{indent}Predict: '{predicted_val}'")
        return

    print(f"{indent}IF Feature {feature_idx} <= {threshold_val}:")
    print(f"{indent}  --> Left (True):")
    print_tree(left_branch, indent + "    ")
    print(f"{indent}  --> Right (False):")
    print_tree(right_branch, indent + "    ")


if __name__ == "__main__":
    data = [
        [6.0, 1.0, 'No'],
        [7.0, 2.0, 'No'],
        [8.0, 3.0, 'No'],
        [6.0, 4.0, 'Yes'],
        [7.0, 5.0, 'Yes'],
        [7.0, 6.0, 'Yes'],
        [8.0, 7.0, 'No'],
        [9.0, 8.0, 'No'],
        [10.0, 9.0, 'Yes'],
        [5.0, 0.5, 'No'],
        [6.5, 3.5, 'Yes'],
        [7.5, 6.5, 'No']
    ]

    target_label_column_index = len(data[0]) - 1

    train_data = data[:10]
    test_data = data[10:]
    y_true_test = [row[target_label_column_index] for row in test_data]
    X_test = [row[:-1] for row in test_data]

    print("--- Building Decision Tree ---")
    decision_tree_root = build_tree(train_data, target_label_column_index, max_depth=3, min_samples_split=2)

    print("\n--- Decision Tree Structure ---")
    print_tree(decision_tree_root)

    print("\n--- Making Predictions ---")
    predictions_on_test_data = predict(decision_tree_root, X_test)

    print("Test True Labels:", y_true_test)
    print("Test Predictions:", predictions_on_test_data)

    acc = accuracy_score(y_true_test, predictions_on_test_data)
    print(f"\nAccuracy on Test Data: {acc:.2f}")

    new_sample = [7.2, 5.8]
    predicted_label = predict_single(decision_tree_root, new_sample)
    print(f"\nPrediction for {new_sample}: {predicted_label}")