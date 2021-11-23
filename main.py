import os
import time

import numpy as np
from sklearn.cluster import KMeans

from FeatureAnalyzer import FeatureAnalyzer


def fetch_user_data(user_id):
    all_signatures = []

    for signature_id in range(1, 41):
        filename = "U" + str(user_id) + "S" + str(signature_id) + ".txt"
        path = os.path.join(os.getcwd(), "SVC2004_Online", filename)
        with open(path) as file:
            lines = file.readlines()[1:]  # discard the first row

            sig = np.zeros(shape=(len(lines), 7), dtype="int32")

            for line_idx, line in enumerate(lines):
                line = line.strip().split()

                for val_idx, val in enumerate(line):
                    sig[line_idx][val_idx] = int(val)

            all_signatures.append(sig)

    return np.asarray(all_signatures, dtype=object)


def split_data_set(signatures, labels, ratios):
    indices = np.random.permutation(len(signatures))
    l = len(signatures)
    valid_start_idx = int(l * ratios[0])
    test_start_idx = int(l * ratios[1])

    X_train = signatures[indices[:valid_start_idx]]
    y_train = labels[indices[:valid_start_idx]]
    X_valid = signatures[indices[valid_start_idx:test_start_idx]]
    y_valid = labels[indices[valid_start_idx:test_start_idx]]
    X_test = signatures[indices[test_start_idx:]]
    y_test = labels[indices[test_start_idx:]]

    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)


def cluster_feature_vectors(number_of_global_features, feature_vectors):
    kmeans = KMeans(n_clusters=int(number_of_global_features * 0.6))
    kmeans.fit(feature_vectors)
    #TODO


def main():
    number_of_global_features = 20

    user_id = input("What is the user id?")
    user_signatures = fetch_user_data(user_id)

    labels = np.array([x // 20 for x in range(40)])  # first 20 valid, second 20 forgery. forgery = 1

    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = split_data_set(user_signatures, labels,
                                                                              ratios=(0.7, 0.15, 0.15))
    analyzer = FeatureAnalyzer()

    signature_features = np.zeros(shape=(len(user_signatures), number_of_global_features))

    print("Creating Feature Vectors")
    for idx, sig in enumerate(user_signatures):
        signature_features[idx] = analyzer.fit_transform(sig)
    print("Feature Vectors Are Ready")

    feature_vectors = signature_features.transpose()

    cluster_feature_vectors(number_of_global_features, feature_vectors)

    with np.printoptions(precision=8, suppress=True):
        print(feature_vectors[0])
        print(feature_vectors[10])
        print(feature_vectors[19])


if __name__ == '__main__':
    main()
