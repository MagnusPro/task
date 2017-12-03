import numpy as np


def read_data(file_name):

    data = np.genfromtxt(file_name, delimiter=',')

    X = data[:, :-1]

    return X


def belonging(X, c):
    b = []
    for x in X:
        d = []
        for centr in c:
            d.append(np.linalg.norm(x - centr))
        b.append(np.argmin(d))
    return np.array(b)


def calc_c(X, b, k):
    c = []
    for i in range(k):
        c.append(X[b == i].mean(axis=0))
    return np.array(c)



def k_means(X, k=3):
    #a lot depends on how to choice first centrs
    #ind = [0, 80, 149]
    ind = np.random.choice(len(X), k, replace=False)

    c = X[ind]

    i = 0

    while True:
        print("iteration {}".format(i))
        b = belonging(X, c)
        c_prev = c
        c = calc_c(X, b, k)
        if np.array_equal(c, c_prev):
            break
        i += 1

    print(belonging(X, c))

X = read_data('iris_data.txt')

k_means(X)

