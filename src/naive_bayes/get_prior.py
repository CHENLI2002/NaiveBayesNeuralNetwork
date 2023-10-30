import os
import numpy as np


def get_data(data_path):
    data = {"english": [], "japanese": [], "spanish": []}

    for folder in os.listdir(data_path):
        for filename in os.listdir(data_path + "/" + folder):
            if int(filename.split(".")[0][1:]) < 10:
                with open(os.path.join(data_path, folder, filename), 'r', encoding='utf8') as f:
                    data[folder].append(f.read().lower())

    return data


def get_prior(data, alpha):
    length = sum(len(folder) for folder in data.values())
    priors = {}
    log_priors = {}
    types = len(data.keys())

    for key in data.keys():
        count = len(data[key])
        prior = (count + alpha) / (length + types * alpha)
        priors[key] = prior
        log_priors[key] = np.log(prior)

    return priors, log_priors
