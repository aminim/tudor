from sklearn.metrics import accuracy_score, f1_score
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from self_learning import SLA
import numpy as np
import time


def read_binary_pendigits(path_to_data):
    df = load_svmlight_file(path_to_data)
    features = df[0].todense().view(type=np.ndarray)
    target = df[1].astype(np.int)
    # classification task is to distinguish between 4 and 9
    condition = np.logical_or((target == 9), (target == 4))
    x = features[condition, :]
    y = target[condition]
    # label is 0, when the image depicts 4, label is 1 otherwise
    y[y == 4] = -1
    y[y == 9] = 1
    return x, y


def split_data(x, y, random_state):
    x_l, x_u, y_l, y_u = train_test_split(x, y, train_size=5, test_size=2189, random_state=random_state)
    print("Number of labeled examples:", x_l.shape)
    print("Number of unlabeled examples:", x_u.shape)
    return x_l, y_l, x_u, y_u


def main():
    x, y = read_binary_pendigits('pendigits')
    x_l, y_l, x_u, y_u = split_data(x, y, random_state=0)

    t0 = time.time()
    model = RandomForestClassifier(n_estimators=200, oob_score=True, random_state=0)
    print("Supervised Random Forest:")
    model.fit(x_l, y_l)
    y_pred = model.predict(x_u)
    print("ACC:", accuracy_score(y_u, y_pred))
    print("F1:", f1_score(y_u, y_pred))
    t1 = time.time()
    print("Time:", t1-t0)

    t0 = time.time()
    print("Self Learning Algorithm:")
    model, thetas = SLA(x_l, y_l, x_u, n_jobs=8, random_state=0)
    y_pred = model.predict(x_u)
    print("ACC:", accuracy_score(y_u, y_pred))
    print("F1:", f1_score(y_u, y_pred))
    t1 = time.time()
    print("Time:", t1-t0)


if __name__ == '__main__':
    main()
