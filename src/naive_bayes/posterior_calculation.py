import numpy as np


# To calculate the posterior, we just need to multiply the probability (the denominator will remain
# constant). In the log space, we need to add those log probabilities and do comparison.
def calculate_predict(log_prior, log_likelihood):
    # print(log_prior)
    # print(log_likelihood)
    list_prior = np.array(list(log_prior.values()))
    list_likelihood = np.array(list(log_likelihood.values()))
    log_result = np.add(list_prior, list_likelihood)
    # print(log_result)
    prediction = list(log_prior.keys())[np.argmax(log_result)]
    return log_result, prediction
