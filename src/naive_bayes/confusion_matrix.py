import os
from src.naive_bayes.get_conditional import calculate_likelihood
from src.naive_bayes.posterior_calculation import calculate_predict


def get_test_data_1(path):
    data = {"english": [], "japanese": [], "spanish": []}

    for folder in os.listdir(path):
        for filename in os.listdir(path + "/" + folder):
            if int(filename.split(".")[0][1:]) >= 10:
                with open(os.path.join(path, folder, filename), 'r', encoding='utf8') as f:
                    data[folder].append(f.read().lower())

    return data


def get_bag_of_character(text):
    characters = "abcdefghijklmnopqrstuvwxyz "
    count = {}

    for char in characters:
        count[char] = 0

    for char in text:
        if char in characters:
            count[char] += 1

    return list(count.values())


def generate_confusion_matrix(log_prior, log_conditional, data):
    result = {"english": {"english": 0, "japanese": 0, "spanish": 0},
              "japanese": {"english": 0, "japanese": 0, "spanish": 0},
              "spanish": {"english": 0, "japanese": 0, "spanish": 0}}

    for key in data.keys():
        for document in data[key]:
            b_o_c = get_bag_of_character(document)
            log_likelihood = calculate_likelihood(log_conditional, b_o_c)
            _, prediction = calculate_predict(log_prior, log_likelihood)
            result[key][prediction] += 1

    print("   (TrueLabel)  English  Spanish  Japanese")
    print(f"English       {result['english']['english']}          {result['spanish']['english']}          {result['japanese']['english']}")
    print(f"Spanish       {result['english']['spanish']}          {result['spanish']['spanish']}          {result['japanese']['spanish']}")
    print(f"Japanese      {result['english']['japanese']}          {result['spanish']['japanese']}          {result['japanese']['japanese']}")
