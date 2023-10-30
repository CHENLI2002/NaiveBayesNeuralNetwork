from src.naive_bayes.get_prior import get_data
import numpy as np


def get_conditional(data_in, alpha):
    characters = "abcdefghijklmnopqrstuvwxyz "
    result_count = {"english": {}, "spanish": {}, "japanese": {}}
    conditionals = {}
    log_conditionals = {}

    for dictionary in result_count.values():
        for char in characters:
            dictionary[char] = 0

    for key in result_count.keys():
        language_data = data_in[key]
        sum_all = 0
        conditional = []
        log_conditional = []

        for document in language_data:
            for char in document:
                if char in characters:
                    result_count[key][char] += 1
                    sum_all += 1
        denominator = sum_all + 27 * alpha

        for char in characters:
            numerator = result_count[key][char] + alpha
            result = numerator / denominator
            conditional.append(result)
            log_conditional.append(np.log(result))

        conditionals[key] = conditional
        log_conditionals[key] = log_conditional

    return conditionals, log_conditionals


def calculate_likelihood(log_conditional, bag_of_character):
    likelihoods = {}
    # print(log_conditional)
    # print(bag_of_character)
    classes = ["english", "japanese", "spanish"]

    for val in classes:
        log_probs = np.array(list(log_conditional[val]))
        bag_of_characters_array = np.array(list(bag_of_character))
        log_likelihood = np.sum(log_probs * bag_of_characters_array)
        likelihoods[val] = log_likelihood

    return likelihoods


if __name__ == "__main__":
    data = get_data("../data")
    cond, log_con = get_conditional(data, 1 / 2)
    print(cond)
    print(log_con)
