import numpy as np

from sklearn.linear_model import LogisticRegression


def read_data(file_name):
    data = np.genfromtxt(file_name, delimiter=', ')

    X = data[:, :-1]

    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

    X = np.hstack((np.ones((data.shape[0], 1)), X))

    y = data[:, -1]

    y[y == -1] = 0

    return X, y

def h(X, w):
    return 1/(1 + np.exp(-np.dot(X, w)))

def cost_func(X, w, y):
    predictions = h(X, w)

    return np.sum(
        -y * np.log(predictions) - (1 - y) * np.log(1 - predictions)
    ) / X.shape[0]


def gradient_cost_func(X, w, y):
    y_p = h(X, w)

    w_grad = np.dot(X.T, (y_p - y)) / X.shape[0]
    return w_grad


def gradient_descent(X, y, a = 0.01):
    w = np.array([1, 0, 0])

    for epoch in range(10000):
        w = w - a * gradient_cost_func(X, w, y)

        if epoch % 1000 == 0:
            print(cost_func(X, w, y))

    return w

def test(X, w, y):
    return np.round(h(X, w))

def algorithm(X_train, X_test, y_train, y_test):
    w = gradient_descent(X_train, y_train)

    y_predicted = test(X_test, w, y_test)

    print("my model", accuracy(np.round(y_predicted.flatten()), y_test))

    return y_predicted

def accuracy(predicted_labels, actual_labels):
    diff = predicted_labels - actual_labels
    return 1.0 - np.count_nonzero(diff) / len(diff)


X_train, y_train = read_data('banana-5-1tra.dat')

X_test, y_test = read_data('banana-5-1tst.dat')

y_p_alg = algorithm(X_train, X_test, y_train, y_test)


model = LogisticRegression()

model.fit(X_train[:, 1:], y_train)

y_p = model.predict(X_test[:, 1:])

print("sklearn", accuracy(np.round(y_p), y_test))

