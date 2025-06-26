import numpy as np
from collections import Counter


def knn(k, data, dataClass, inputs):
    nInputs = np.shape(inputs)[0]
    closest = np.zeros(nInputs)

    for n in range(nInputs):
        distances = np.sum((data - inputs[n, :])**2, axis=1)
        indices = np.argsort(distances, axis=0)
        k_nearest_classes = dataClass[indices[:k]]
        class_counts = Counter(k_nearest_classes)
        most_common = class_counts.most_common(1)

        if most_common:
            closest[n] = most_common[0][0]
        else:
            closest[n] = dataClass[indices[0]]

    return closest


data = np.array([
    [1, 1],
    [1.5, 1.5],
    [5, 5],
    [5.5, 5.5],
    [1, 5],
    [5, 1]
])

dataClass = np.array([0, 0, 1, 1, 0, 1])

inputs = np.array([
    [1.2, 1.2],
    [5.2, 5.2],
    [3, 3]
])

k_value = 3
predicted_classes = knn(k_value, data, dataClass, inputs)
print("Predicted classes for inputs:", predicted_classes)
