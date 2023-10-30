import numpy as np
import torch
import torchvision.transforms
from torch.utils.data import Dataset
from torchvision import datasets
from src.nn_source import forward_pass, backward_pass
from matplotlib import pyplot

if __name__ == "__main__":
    X_SIZE = 784
    OUT_PUT_1_SIZE = 300
    BATCH_SIZE = 128
    OUT_PUT_2_SIZE = 10

    train_data = datasets.MNIST(root="data/nn/mnist_train", train=True, download=True,
                                transform=torchvision.transforms.ToTensor())
    test_data = datasets.MNIST(root="data/nn/mnist_test", train=False, download=True,
                               transform=torchvision.transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    learning_rate = 0.1
    epochs = 5
    loss_over_time = []

    W1 = (np.random.rand(X_SIZE, OUT_PUT_1_SIZE) * 2 - 1) * 0.1
    W2 = (np.random.rand(OUT_PUT_1_SIZE, OUT_PUT_2_SIZE) * 2 - 1) * 0.1

    for e in range(epochs):
        for image, label in train_loader:
            input_data = np.array(image.reshape(-1, X_SIZE))
            np_label = np.array(label)
            actual_size = np.shape(input_data)[0]
            new_dw1 = 0
            new_dw2 = 0
            sum_of_loss = 0

            for i in range(actual_size):
                # print(np.shape(input_data))
                current_input = np.array(input_data[i]).reshape(784, 1)
                # print(np.shape(current_input))
                current_label = np_label[i]
                # print(current_label)
                l_one_hot = np.eye(10)[current_label]
                fp = forward_pass(W1, W2, current_input)
                temp_dw1, temp_dw2, loss = backward_pass(l_one_hot, W2, current_input, fp)
                new_dw1 += temp_dw1
                new_dw2 += temp_dw2
                sum_of_loss += loss

            new_dw1 = new_dw1 / actual_size
            new_dw2 = new_dw2 / actual_size
            W1 = W1 - learning_rate * new_dw1
            W2 = W2 - learning_rate * new_dw2
            avg_loss = sum_of_loss / actual_size
            loss_over_time.append(avg_loss)
            print(f"Average loss of this batch is: {avg_loss}")

    total_num = 0
    err = 0

    # Testing for error
    for image, label in test_loader:
        input_data = np.array(image.reshape(-1, X_SIZE))
        actual_size = np.shape(input_data)[0]
        np_label = np.array(label)
        for i in range(actual_size):
            current_input = np.array(input_data[i]).reshape(784, 1)
            current_label = np_label[i]
            l_one_hot = np.eye(10)[current_label]
            fp = forward_pass(W1, W2, current_input)
            prediction = np.argmax(fp["softmax"])
            if prediction != current_label:
                err += 1

            total_num += 1

    print(f"Test Error is {err/total_num}")

    pyplot.figure(figsize=(10, 10))
    pyplot.plot(loss_over_time, c="r")
    pyplot.ylabel("cross-entropy-loss")
    pyplot.xlabel("time")
    pyplot.title("Learning Curve based on Cross-Entropy-Loss")
    pyplot.savefig("LearningCurve.png")
    pyplot.show()
