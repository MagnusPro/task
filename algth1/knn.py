import numpy as np
from scipy import stats
from sklearn.neighbors import KNeighborsClassifier

def read_data(file_name):

    labels = [b'Iris-setosa', b'Iris-versicolor', b'Iris-virginica']
    data = np.genfromtxt(file_name, delimiter=',', converters = {4: lambda s: np.float(labels.index(s))})

    return data[:, :-1], data[:, -1]

def train_test_split(X, y, train_part=0.67):
    train = int(X.shape[0] * train_part)

    index = np.random.permutation(X.shape[0])

    X_train = X[index[:train], :]
    X_test = X[index[train:], :]

    y_train = y[index[:train]]
    y_test = y[index[train:]]

    return X_train, y_train, X_test, y_test


def calc_distances(X_train, X_test):
    return np.sqrt(((X_test - X_train[:, np.newaxis]) ** 2).sum(axis=2))

def predict_class(y_train, D, k):
    y_pred = []
    for dist in D.T:
        index = dist.argsort()
        y_pred.append(stats.mode(y_train[index[:k]]).mode)
    return np.array(y_pred)

def accuracy(y_pred, y_test):
    count = 0
    for i in range(y_pred.shape[0]):
        if y_pred[i] == y_test[i]:
            count += 1
    return count / y_pred.shape[0] * 100


def algorithm(k=3):
    X, y = read_data('iris_data.txt')

    X_train, y_train, X_test, y_test = train_test_split(X, y, 0.67)

    D = calc_distances(X_train, X_test)

    y_pred = predict_class(y_train, D, k)

    print('my alg', accuracy(y_pred, y_test))

    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X_train, y_train)

    ski_y_pred = neigh.predict(X_test)

    print('skit alg', accuracy(ski_y_pred, y_test))

algorithm()
