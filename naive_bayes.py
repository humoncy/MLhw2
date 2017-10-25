import numpy as np
import matplotlib.pyplot as plt
import sys
from mnist import MNIST


def show_digits(img, label):
    """
    Show 0~9 digit in one image.
    :param img: mnist image
    :param label: mnist label
    :return:
    """
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


def cal_probabilities_discrete(train_img, train_label):
    """
    Calculate probabilities of each pixel.
    Tally the frequency of the values of each pixel into 32 bins.
    :param train_img:
    :param train_label:
    :return:
        p: 10 * (28*28) * 32 - probability of the value of each pixel
        pi: 10 * 1 - frequency of labels
    """
    p = np.zeros((10, 784, 32))
    pi = np.zeros(10)
    for i in range(len(train_img)):
        print("Training %dth image." % i)
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
                    p[i, j, k] = 0.01
            p[i, j] /= sum(p[i, j])

    return pi, p


def cal_probabilities_countinuous(train_img, train_label):
    """
    Calculate probabilities of each pixel.
    Tally the frequency of the values of each pixel.
    :param train_img:
    :param train_label:
    :return:
        p: 10 * (28*28) * 256 - probability of the value of each pixel
        pi: 10 * 1 - frequency of labels
    """
    p = np.ones((10, 784, 256))
    pi = np.zeros(10)
    for i in range(len(train_img)):
        print("Training %dth image." % i)
        l = train_label[i]
        pi[l] += 1
        for j in range(784):
            v = train_img[i][j]
            p[l, j, v] += 1
    pi /= len(train_img)
    for i in range(10):
        for j in range(784):
            p[i, j] /= sum(p[i, j])

    return pi, p


def mean_stdev(p):
    """
    Mean and standard deviation of pixel values (0~255) of each labels.
    :param p:
    :return:
        mean: 10 * 784
        stdev: 10 * 784
    """
    mean = np.zeros((10, 784))
    stdev = np.zeros((10, 784))
    num_bins = len(p[0][0])
    x = np.arange(len(p[0][0]))
    bin_size = 256 / num_bins
    x = x * bin_size + ((bin_size - 1) / 2)

    for i in range(10):
        for j in range(784):
            mean[i, j] = np.dot(x, p[i, j])
            stdev[i, j] = max(8.0, np.sqrt(np.dot(np.square(x), p[i, j]) - np.power(mean[i, j], 2)))
    return mean, stdev


def gaussian_probability(mean, stdev, x):
    """
    Calculate Gaussian probability.
    """
    exponent = np.exp(-(np.power(x - mean, 2) / (2 * np.power(stdev, 2))))
    return (1 / (np.sqrt(2 * np.pi) * stdev)) * exponent


def naive_bayes_discrete(pi, p, test_img):
    """
    Perform Naive Bayes classifier.
    :param pi: frequency of labels
    :param p: frequency of pixels of labels
    :param test_img: test image
    :return: predictions of each test image
    """
    predictions = np.zeros(len(test_img))
    posteriori = np.zeros(len(test_img))
    for i in range(len(test_img)):
        print("Testing %dth image." % i)
        log_probabilities = np.zeros(10)
        for k in range(10):
            ''' in log scale to avoid underflow '''
            log_probabilities[k] += np.log(pi[k])
            for j in range(784):
                v = test_img[i][j]
                b = v // 8
                log_probabilities[k] += np.log(p[k, j, b])
        posteriori[i] = max(log_probabilities)
        predictions[i] = np.argmax(log_probabilities)
    print("posteriori: ")
    print(posteriori)
    return predictions


def naive_bayes_continuous(pi, mean, stdev, test_img):
    predictions = np.zeros(len(test_img))
    max_posteriori = np.zeros(len(test_img))
    for i in range(len(test_img)):
        print("Testing %dth image." % i)
        log_probabilities = np.zeros(10)
        for k in range(10):
            log_probabilities[k] += np.log(pi[k])
            for j in range(784):
                log_probabilities[k] += np.log(gaussian_probability(mean[k, j], stdev[k, j], test_img[i][j]))
        max_posteriori[i] = max(log_probabilities)
        predictions[i] = np.argmax(log_probabilities)
    print("posteriori: ")
    print(max_posteriori)
    return predictions


def cal_error_rate(predictions, test_label):
    num_correct = 0
    for i in range(len(predictions)):
        if predictions[i] == test_label[i]:
            num_correct += 1
    print("number of correct/total: %d/%d" % (num_correct, len(test_label)))
    error_rate = (len(test_label) - num_correct) / len(test_label)
    print("error_rate: %f％" % (error_rate * 100))
    return error_rate


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: ", sys.argv[0], "<toggle_option (0 or 1)>")
        print("Use default toggle option 0.")
        toggle_option = 0
    else:
        toggle_option = int(sys.argv[1])

    mndata = MNIST('/Users/islab/PycharmProjects/ml/hw2/data')
    train_img, train_label = mndata.load_training()
    test_img, test_label = mndata.load_testing()
    # show_digits(train_img, train_label)

    if toggle_option == 0:
        pi, p = cal_probabilities_discrete(train_img, train_label)
        predictions = naive_bayes_discrete(pi, p, test_img)
    else:
        pi, p = cal_probabilities_discrete(train_img, train_label)
        # pi, p = cal_probabilities_countinuous(train_img, train_label)
        mean, stdev = mean_stdev(p)
        predictions = naive_bayes_continuous(pi, mean, stdev, test_img)

    cal_error_rate(predictions, test_label)
