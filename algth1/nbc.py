import numpy as np
from sklearn.naive_bayes import GaussianNB


def read_data(file_name):

    labels = [b'Iris-setosa', b'Iris-versicolor', b'Iris-virginica']
    data = np.genfromtxt(file_name, delimiter=',', converters = {4: lambda s: np.float(labels.index(s))})

    return data


def train_test_split(X, train_part=0.67):
    train = int(X.shape[0] * train_part)

    index = np.random.permutation(X.shape[0])

    X_train = X[index[:train], :]
    X_test = X[index[train:], :]

    return X_train, X_test


def separateByClass(dataset):
    return {
        label: dataset[dataset[:, -1] == label, :]
        for label in np.unique(dataset[:, -1])
    }


def summarize(dataset, N):
    means = dataset.mean(axis=0)[:-1]
    stds = dataset.std(axis=0, ddof=1)[:-1]

    # add probability of class
    p = dataset.shape[0] / N

    return means, stds, p


def summarizeByClass(dataset):
    separated = separateByClass(dataset)

    summaries = {}

    for classValue, instances in separated.items():
        summaries[classValue] = summarize(instances, dataset.shape[0])


    return summaries


def calculateFeaturesProbability(x, mean, stdev):
    return np.exp(-(x - mean) ** 2 / (2 * stdev ** 2)) / (np.sqrt(2 * np.pi) * stdev)


def calculateProb(summaries, inputVector):
    probabilities = {}

    for classValue, classSummaries in summaries.items():
        means = classSummaries[0]
        stds = classSummaries[1]
        #use probability of class
        p = classSummaries[2]

        # Calculate probability of vector
        probabilities[classValue] = np.prod(calculateFeaturesProbability(inputVector[:-1], means, stds)) * p

    return probabilities


def predictLabel(summaries, inputVector):
    # Calculate probabilities
    probabilities = calculateProb(summaries, inputVector)

    # Init values of probability and label
    bestLabel, bestProb = None, -1

    # Check probability of which class is better
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue

    return bestLabel


def predict(summary, test_data):
    predictions = []

    for i in range(len(test_data)):
        result = predictLabel(summary, test_data[i])
        predictions.append(result)

    return np.array(predictions)

def accuracy(y_pred, y_test):
    count = 0
    for i in range(y_pred.shape[0]):
        if y_pred[i] == y_test[i]:
            count += 1
    return count / y_pred.shape[0] * 100


def algorithm():
    data = read_data('iris_data.txt')
    data_train, data_test = train_test_split(data)

    summary = summarizeByClass(data_train)

    y_pred = predict(summary, data_test)

    print('my_alg', accuracy(y_pred, data_test[:, -1]))

    gnb = GaussianNB()
    y_pred_2 = gnb.fit(data_train[:, :-1], data_train[:, -1]).predict(data_test[:, :-1])

    print('skitlearn alg', accuracy(y_pred_2, data_test[:, -1]))

algorithm()

