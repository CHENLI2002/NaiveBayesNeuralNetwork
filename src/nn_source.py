import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def soft_max(z):
    # assuming z is a vector
    expo = np.exp(z - np.max(z))
    return expo / np.sum(expo)


def forward_pass(w1, w2, x):
    z1 = np.dot(x.T, w1)
    a1 = sigmoid(z1)
    # print(np.shape(a1))
    z2 = np.dot(a1, w2)
    y_hat = soft_max(z2)

    return {"z1": z1, "a1": a1, "z2": z2, "softmax": y_hat}


def backward_pass(y, w2, x, forward_pass_in):
    cross_entropy_loss = -np.sum(y * np.log(forward_pass_in["softmax"]))
    dz2 = forward_pass_in["softmax"] - y
    # print(np.shape(dz2))
    dw2 = np.dot(forward_pass_in["a1"].T, dz2)
    da1 = np.dot(dz2, w2.T)
    sigmoid_z1 = sigmoid(forward_pass_in["z1"])
    sigmoid_d = sigmoid_z1 * (1 - sigmoid_z1)
    # print(np.shape(da1))
    # print(np.shape(sigmoid_d))
    # print(np.shape(x))

    dw1 = np.dot(x, (da1 * sigmoid_d))
    # print(f"dw1 is {dw1}")
    # print(f"dw2 is {dw2}")

    return dw1, dw2, cross_entropy_loss
