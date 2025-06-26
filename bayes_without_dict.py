import math
from collections import Counter


def _calculate_feature_stats(data):

    if not data:
        return []

    transposed_data = list(zip(*data))

    means = [sum(col) / len(col) for col in transposed_data]

    stds = []
    for i, col_data in enumerate(transposed_data):
        if len(col_data) > 1:
            variance = sum([(x - means[i]) ** 2 for x in col_data]) / (len(col_data) - 1)
            std = math.sqrt(variance)
        else:
            std = 0.0
        stds.append(std if std > 1e-6 else 1e-6)

    return [{'mean': m, 'std': s} for m, s in zip(means, stds)]


def _gaussian_pdf(x, mean, std):

    if std == 0:
        return 1.0 if x == mean else 0.0
    exponent = -((x - mean) ** 2) / (2 * (std ** 2))
    return (1 / (math.sqrt(2 * math.pi) * std)) * math.exp(exponent)


def fit_naive_bayes(x, y):

    classes = list(set(y))

    class_counts = Counter(y)
    class_priors = {label: count / len(y) for label, count in class_counts.items()}

    separated_x = {label: [] for label in classes}
    for i, row in enumerate(x):
        separated_x[y[i]].append(row)

    feature_stats = {
        label: _calculate_feature_stats(data_for_class)
        for label, data_for_class in separated_x.items()
    }

    return classes, class_priors, feature_stats


def predict_proba_naive_bayes(x_row, classes, class_priors, feature_stats):

    probabilities = {}
    for class_label in classes:
        prior = class_priors[class_label]
        likelihood = 1.0

        stats = feature_stats[class_label]
        for i, feature_value in enumerate(x_row):
            likelihood *= _gaussian_pdf(feature_value, stats[i]['mean'], stats[i]['std'])

        probabilities[class_label] = prior * likelihood

    sum_probs = sum(probabilities.values())
    if sum_probs == 0:
        return {cls: 1 / len(classes) for cls in classes}

    return {cls: prob / sum_probs for cls, prob in probabilities.items()}


def predict_naive_bayes(x, classes, class_priors, feature_stats):

    predictions = []
    for x_row in x:
        probabilities = predict_proba_naive_bayes(x_row, classes, class_priors, feature_stats)

        if not probabilities:
            predictions.append(None)
        else:
            best_class = max(probabilities, key=probabilities.get)
            predictions.append(best_class)
    return predictions


if __name__ == "__main__":
    x_train = [
        [5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2], [4.7, 3.2, 1.3, 0.2],
        [7.0, 3.2, 4.7, 1.4], [6.4, 3.2, 4.5, 1.5], [6.9, 3.1, 4.9, 1.5],
        [6.3, 3.3, 6.0, 2.5], [5.8, 2.7, 5.1, 1.9], [7.1, 3.0, 5.9, 2.1],
        [5.0, 3.4, 1.5, 0.2], [5.4, 3.9, 1.7, 0.4],
        [6.1, 2.8, 4.0, 1.3], [6.0, 2.9, 4.5, 1.5],
        [6.3, 2.5, 5.0, 1.9]
    ]
    y_train = [
        'Sentosa', 'Sentosa', 'Sentosa',
        'Vesicolor', 'Vesicolor', 'Vesicolor',
        'Virginia', 'Virginia', 'Virginia',
        'Sentosa', 'Sentosa',
        'Vesicolor', 'Vesicolor',
        'Virginia'
    ]

    print("--- Training the Naive Bayes Classifier ---")
    classes_learned, class_priors_learned, feature_stats_learned = fit_naive_bayes(x_train, y_train)

    print("\nTraining complete. Model statistics:")
    print("Class Priors:", class_priors_learned)
    print("Feature Stats (Mean, Std):", feature_stats_learned)

    print("\n--- Making Predictions ---")
    test_samples = [
        [5.2, 3.4, 1.5, 0.2],
        [6.2, 3.0, 4.8, 1.8],
        [7.2, 3.6, 6.1, 2.5],
        [5.5, 2.3, 4.0, 1.3]
    ]

    predictions = predict_naive_bayes(test_samples, classes_learned, class_priors_learned, feature_stats_learned)
    print("\nTest Samples:")
    for i, sample in enumerate(test_samples):
        print(f"Sample {i + 1}: {sample} -> Predicted Class: {predictions[i]}")

    print("\nProbabilities for Sample 2 ([6.2, 3.0, 4.8, 1.8]):")
    probabilities_sample_2 = predict_proba_naive_bayes([6.2, 3.0, 4.8, 1.8], classes_learned, class_priors_learned,
                                                       feature_stats_learned)
    for class_name, prob in probabilities_sample_2.items():
        print(f"  P({class_name}|X) = {prob:.4f}")

    train_predictions = predict_naive_bayes(x_train, classes_learned, class_priors_learned, feature_stats_learned)
    correct_count = sum(1 for i in range(len(x_train)) if train_predictions[i] == y_train[i])
    accuracy = correct_count / len(x_train)
    print(f"\nTraining Accuracy: {accuracy:.2f}")
