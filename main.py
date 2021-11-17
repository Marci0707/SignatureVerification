import os

import numpy as np


def fetch_user_data(user_id):
    all_signatures = []
    for signature_id in range(1, 41):
        signature = []
        filename = "U" + str(user_id) + "S" + str(signature_id) + ".txt"
        path = os.path.join(os.getcwd(), "SVC2004_Online", filename)
        with open(path) as file:
            lines = file.readlines()[1:]  # discard the first row
            for line in lines:
                line = line.strip().split()
                data_point = tuple([int(x) for x in line])
                signature.append(data_point)

            all_signatures.append(signature)

    return all_signatures

def split_data_set(signatures, labels, ratios):
    indices = np.random.permutation(len(signatures))
    l = len(signatures)
    valid_start_idx = int(l*ratios[0])
    test_start_idx = int(l*ratios[1])

    X_train = signatures[indices[:valid_start_idx]]
    y_train = labels[indices[:valid_start_idx]]
    X_valid = signatures[indices[valid_start_idx:test_start_idx]]
    y_valid = labels[indices[valid_start_idx:test_start_idx]]
    X_test = signatures[indices[test_start_idx:]]
    y_test = labels[indices[test_start_idx:]]

    return (X_train,y_train), (X_valid,y_valid),(X_test,y_test)




def main():
    user_id = input("What is the user id?")
    user_signatures = fetch_user_data(user_id)
    labels = [(x-1)//20 for x in range(1,40)]
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = split_data_set(user_signatures,labels, ratios=(0.7, 0.15, 0.15))



if __name__ == '__main__':
    main()
