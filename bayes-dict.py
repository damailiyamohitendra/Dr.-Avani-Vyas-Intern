from collections import defaultdict


def train_naive_bayes(data):
    model = {}
    class_counts = defaultdict(int)
    feature_counts = defaultdict(lambda: defaultdict(int))

    for features, label in data:
        class_counts[label] += 1
        for feature in features:
            feature_counts[label][feature] += 1

    total_samples = sum(class_counts.values())
    for label in class_counts:
        model[label] = {'prior': class_counts[label]/total_samples, 'features': {}}
        total_label_features = sum(feature_counts[label].values())
        for feature in feature_counts[label]:
            model[label]['features'][feature] = feature_counts[label][feature]/total_label_features

    return model


def predict_naive_bayes(model, features):
    best_label = None
    max_prob = -1

    for label in model:
        prob = model[label]['prior']
        for feature in features:
            prob *= model[label]['features'].get(feature, 1e-6)
        if prob > max_prob:
            max_prob = prob
            best_label = label

    return best_label


dataset = [
    (['sunny', 'hot', 'high', 'weak'], 'no'),
    (['sunny', 'hot', 'high', 'strong'], 'no'),
    (['overcast', 'hot', 'high', 'weak'], 'yes'),
    (['rain', 'mild', 'high', 'weak'], 'yes'),
    (['rain', 'cool', 'normal', 'weak'], 'yes'),
    (['rain', 'cool', 'normal', 'strong'], 'no'),
    (['overcast', 'cool', 'normal', 'strong'], 'yes'),
    (['sunny', 'mild', 'high', 'weak'], 'no'),
    (['sunny', 'cool', 'normal', 'weak'], 'yes'),
    (['rain', 'mild', 'normal', 'weak'], 'yes'),
    (['sunny', 'mild', 'normal', 'strong'], 'yes'),
    (['overcast', 'mild', 'high', 'strong'], 'yes'),
    (['rain', 'mild', 'high', 'strong'], 'no'),
    (['overcast', 'hot', 'normal', 'weak'], 'yes'),
]

model = train_naive_bayes(dataset)
query_features = ['rain', 'cool', 'normal', 'weak']
prediction = predict_naive_bayes(model, query_features)
print("Predicted Class Label: ", prediction)
