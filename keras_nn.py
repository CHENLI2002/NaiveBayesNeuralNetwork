# The following code is using Keras DNN model.
import keras
import numpy as np
import tensorflow
from torchvision import datasets
from matplotlib import pyplot

if __name__ == "__main__":
    train_data = datasets.MNIST(root="data/nn/mnist_train", train=True, download=True)
    test_data = datasets.MNIST(root="data/nn/mnist_test", train=False, download=True)

    train_image = train_data.data.numpy()
    train_label = train_data.targets.numpy()
    test_image = test_data.data.numpy()
    test_label = test_data.targets.numpy()

    # Initial zero weights
    # initial_weight_1_0 = [np.zeros((28*28, 300)), np.zeros(300)]
    # initial_weight_2_0 = [np.zeros((300, 10)), np.zeros(10)]

    # Initial Random weights (rand produce 0~1)
    # ini_weight_1_rand = [np.random.rand(28*28, 300) * 2 - 1, np.random.rand(300) * 2 - 1]
    # ini_weight_2_rand = [np.random.rand(300, 10) * 2 - 1, np.random.rand(10) * 2 - 1]

    keras_model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(300, activation='sigmoid'),
        keras.layers.Dense(10, activation='softmax'),
    ])

    keras_model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = keras_model.fit(train_image, train_label, epochs=5, batch_size=64)
    test_loss, test_accuracy = keras_model.evaluate(test_image, test_label, verbose=-1)
    print(f"The test error is {1 - test_accuracy}")
    loss_over_time = history.history['loss']

    pyplot.figure(figsize=(10, 10))
    pyplot.plot(loss_over_time, c='r')
    pyplot.ylabel('cross_entropy_loss')
    pyplot.xlabel('training_step')
    pyplot.title('Keras DNN Learning Curve using Loss')
    pyplot.savefig('KerasNNLearningCurve')
    pyplot.show()

