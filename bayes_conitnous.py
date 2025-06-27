import math

# Dataset
x = [
    [2.5, 50, 15],
    [2.7, 50, 60],
    [1.8, 100, 45],
    [2.8, 23, 2]
]

y = ['good', 'bad', 'good', 'worst']

def prob(y, category):
    count = len(y)
    count_category = y.count(category)
    prob_of_category = count_category / count
    return prob_of_category

#  mean 
def mean(x, y, category, index_feature_of_category):
    total = 0
    count = 0
    for i in range(len(y)):
        if y[i] == category:
            total = total + x[i][index_feature_of_category]
            count = count + 1

    mean_value = total / count
    return mean_value

# standard deviation
def standard_deviation(x, y, category, index_feature_of_category):
    mean_value = mean(x, y, category, index_feature_of_category)
    sum_of_squares_of_variance = 0
    count = 0
    for i in range(len(y)):
        if y[i] == category:
            difference = x[i][index_feature_of_category] - mean_value
            square = difference * difference
            sum_of_squares_of_variance = sum_of_squares_of_variance + square
            count = count + 1

    variance = sum_of_squares_of_variance / count
    sd = math.sqrt(variance)
    return sd

# Gaussian probability formula
def formula(x, mean, sd):
    if sd == 0:
        if x == mean:
            return 1
        else:
            return 0

    exponent = math.exp(-((x - mean) ** 2) / (2 * sd * sd))
    other_half = 1 / (math.sqrt(2 * math.pi) * sd)
    result = exponent * other_half
    return result

# Prediction function
def predict(x, y, input):
    unique_labels = []
    for i in y:
        if i not in unique_labels:
            unique_labels.append(i)

    class_probabilities = {}

    for class_name in unique_labels:
        prior = prob(y, class_name)
        likelihood = 1

        for i in range(len(input)):
            mean_value = mean(x, y, class_name, i)
            sd = standard_deviation(x, y, class_name, i)
            feature_prob = formula(input[i], mean_value, sd)
            likelihood = likelihood * feature_prob

        total_prob = prior * likelihood
        class_probabilities[class_name] = total_prob

    best_class = None
    best_prob = -1
    for class_name in class_probabilities:
        if class_probabilities[class_name] > best_prob:
            best_prob = class_probabilities[class_name]
            best_class = class_name

    return best_class, best_prob


test_data = [2.6, 52, 45]
best_c, best_p = predict(x, y, test_data)

# Output
print("Predicted Class:", best_c)
print("Probability:", best_p)
