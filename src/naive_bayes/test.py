def get_test_data(path):
    with open(path, 'r', encoding='utf8') as f:
        data = f.read().lower()

    return data


def present_test(data_path):
    characters = "abcdefghijklmnopqrstuvwxyz "
    count = {}

    for char in characters:
        count[char] = 0

    data = get_test_data(data_path)

    for char in data:
        if char in characters:
            count[char] += 1

    return count.values()


if __name__ == "__main__":
    print(present_test("../../data/bayes/english/e10.txt"))
