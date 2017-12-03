import numpy as np

def load_file(file_name):
    data = np.genfromtxt(file_name, delimiter=',')
    X = data[:, :-1]
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    return X, data[:, -1]


def train_test_split(X, y, train_part=0.67):
    train = int(X.shape[0] * train_part)

    index = range(X.shape[0])

    X_train = X[index[:train], :]
    X_test = X[index[train:], :]

    y_train = y[index[:train]]
    y_test = y[index[train:]]

    return X_train, y_train, X_test, y_test

def square_error(y_p, y_t):
    return np.sum((y_p - y_t) ** 2) / y_t.shape[0]

def gradient_square_error(X, w, y_t):
    y_p = np.dot(w, X.T)
    return 2 * np.dot((y_p - y_t), X)/ y_t.shape[0]

def gradient_descent(X, y, a = 0.1):
    w = np.zeros((X.shape[1],))
    i = 0
    square_error(np.dot(w, X.T), y)
    while i < 1000:
        w = w - a * gradient_square_error(X, w, y)
        print(square_error(np.dot(w, X.T), y))
        i += 1
    return w


def algorithm():
    X, y = load_file('plastic.dat')

    #split data
    X_train, y_train, X_test, y_test = train_test_split(X, y)

    #get regression params
    w = gradient_descent(X_train, y_train)

    #build regression with sklearn
    from sklearn.linear_model import LinearRegression

    m = LinearRegression()

    m.fit(X_train, y_train.reshape((-1, )))

    y_p = m.predict(X=X_test)

    #results on test
    print('my alg', square_error(np.dot(w, X_test.T), y_test))
    print('skitlearn alg', square_error(y_p, y_test))

algorithm()

