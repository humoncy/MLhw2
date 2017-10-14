import numpy as np
import matplotlib.pyplot as plt
import math
import sys
from mnist import MNIST


def show_digits(img, label):
    plt.figure(num='mnist', figsize=(14, 6))
    for i in range(10):
        j = 0
        while train_label[j] != i:
            j = j + 1
        curr_img = np.reshape(train_img[j], (28, 28))
        curr_label = train_label[j]
        plt.subplot(2, 5, i + 1)
        plt.title("Label is " + str(curr_label))
        plt.imshow(curr_img, cmap=plt.get_cmap('gray'))
    plt.show()


def cal_probabilities(train_img, train_label):
    p = np.zeros((10, 784, 32))
    pi = np.zeros(10)
    for i in range(len(train_img)):
        l = train_label[i]
        pi[l] += 1
        for j in range(784):
            # calculate which bin
            b = train_img[i][j] // 8
            p[l, j, b] += 1
    pi /= len(train_img)

    for i in range(10):
        for j in range(784):
            for k in range(32):
                # pseudo count
                if p[i, j, k] == 0:
                    p[i, j, k] = 1
            p[i, j] /= sum(p[i, j])

    return pi, p


def mean_stdev(p):
    mean = np.zeros((10, 784))
    stdev = np.zeros((10, 784))
    x = np.arange(32)
    x = x * 8 + 3.5
    for i in range(10):
        for j in range(784):
            mean[i, j] = np.dot(x, p[i, j])
            stdev[i, j] = math.sqrt(np.dot(np.square(x), p[i, j]) - math.pow(mean[i, j], 2))
    return mean, stdev


def gaussian_probability(mean, stdev, x):
    exponent = math.exp(- math.pow(x - mean, 2) / (2 * math.pow(stdev, 2)))
    return math.pow(math.sqrt(2 * math.pi) * stdev, -1) * exponent


def naive_bayes_discrete(pi, p, test_img):
    predictions = np.zeros(len(test_img))
    posteriori = np.zeros(len(test_img))
    for i in range(len(test_img)):
        log_probabilities = np.zeros(10)
        for k in range(10):
            log_probabilities[k] += math.log(pi[k])
            for j in range(784):
                v = test_img[i][j]
                b = v // 8
                log_probabilities[k] += math.log(p[k, j, b])
        # print(log_probabilities)
        posteriori[i] = max(log_probabilities)
        predictions[i] = np.argmax(log_probabilities)
    # print("posteriori: ")
    # print(posteriori)
    print("predictions: ")
    print(predictions)
    return predictions


def naive_bayes_continuous(pi, mean, stdev, test_img):
    predictions = np.zeros(len(test_img))
    max_posteriori = np.zeros(len(test_img))
    for i in range(len(test_img)):
        log_probabilities = np.zeros(10)
        for k in range(10):
            log_probabilities[k] += math.log(pi[k])
            for j in range(784):
                log_probabilities[k] += math.log(gaussian_probability(mean[k, j], stdev[k, j], test_img[i][j]))
        # print(log_probabilities)
        max_posteriori[i] = max(log_probabilities)
        predictions[i] = np.argmax(log_probabilities)
    # print("posteriori: ")
    # print(max_posteriori)
    print("predictions: ")
    print(predictions)
    return predictions


def cal_error_rate(predictions, test_label):
    print("test_label")
    print(test_label)
    num_correct = 0
    for i in range(len(predictions)):
        if predictions[i] == test_label[i]:
            num_correct += 1
    error_rate = (len(test_label) - num_correct) / len(test_label)
    print("error_rate: %f％" % (error_rate * 100))
    return error_rate


if len(sys.argv) < 2:
    print("Usage: ", sys.argv[0], "<toggle_option (0 or 1)>")
    print("Use default toggle option 0.")
toggle_option = int(sys.argv[1])

mndata = MNIST('/Users/islab/PycharmProjects/ml/hw2/data')
train_img, train_label = mndata.load_training()
test_img, test_label = mndata.load_testing()
# show_digits(train_img, train_label)

pi, p = cal_probabilities(train_img, train_label)

if toggle_option == 0:
    predictions = naive_bayes_discrete(pi, p, test_img)
else:
    mean, stdev = mean_stdev(p)
    predictions = naive_bayes_continuous(pi, mean, stdev, test_img)

cal_error_rate(predictions, test_label)
