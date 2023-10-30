from src.naive_bayes.get_prior import get_data, get_prior
from src.naive_bayes.get_conditional import get_conditional, calculate_likelihood
from src.naive_bayes.test import present_test
from src.naive_bayes.posterior_calculation import calculate_predict
from src.naive_bayes.confusion_matrix import generate_confusion_matrix, get_test_data_1

if __name__ == "__main__":
    alpha = 0.5
    divider = "=========================================================="
    data = get_data("data/bayes")
    presentable_priors, log_priors = get_prior(data, alpha)

    for key in presentable_priors.keys():
        print(f"The prior probability of {key} is {presentable_priors[key]}")

    print(divider)

    conditional, log_conditional = get_conditional(data, alpha)

    for key in conditional.keys():
        print(f"The conditional array of language {key} is {conditional[key]}")

    print(divider)

    b_o_c_test = present_test("data/bayes/english/e10.txt")

    print(f"The bag of characters array of e10.txt is: {b_o_c_test}")
    print(divider)

    log_likelihood = calculate_likelihood(log_conditional, b_o_c_test)
    true_likelihood = {}

    for key in log_likelihood.keys():
        log = log_likelihood[key]
        true_likelihood[key] = "e^(" + str(log) + ")"

    print(true_likelihood)
    print(divider)

    log_result, prediction = calculate_predict(log_priors, log_likelihood)

    printable = ""

    classes = ["english", "japanese", "spanish"]

    for index, item in enumerate(log_result):
        printable += f"Probability of {classes[index]} is " + "e^" + str(item) + "\n"

    print(printable)

    print(f"The prediction is {prediction}")
    print(divider)

    test_data = get_test_data_1("data/bayes")
    generate_confusion_matrix(log_priors, log_conditional, test_data)
