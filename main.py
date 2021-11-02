import os


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


def main():
    user_id = input("What is the user id?")
    user_signatures = fetch_user_data(user_id)
    print(len(user_signatures))


if __name__ == '__main__':
    main()
