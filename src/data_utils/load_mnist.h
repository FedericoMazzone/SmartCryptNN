#include <cstdint>
#include <vector>



/**
 * Load MNIST dataset from data/mnist.txt.
 * @param trainX vector to store the features of training data
 * @param trainY vector to store the labels of training data
 * @param testX vector to store the features of test data
 * @param testY vector to store the labels of test data
 */
void loadMNIST(
    std::vector<std::vector<int64_t>> &trainX,
    std::vector<int64_t> &trainY,
    std::vector<std::vector<int64_t>> &testX,
    std::vector<int64_t> &testY
);


/**
 * Load MNIST dataset from data/mnist.txt.
 * @param trainX vector to store the features of training data
 * @param trainY vector to store the labels of training data
 * @param testX vector to store the features of test data
 * @param testY vector to store the labels of test data
 * @param trainSize how many train samples to load
 * @param testSize how many test samples to load
 */
void loadMNIST(
    std::vector<std::vector<double>> &trainX,
    std::vector<std::vector<double>> &trainY,
    std::vector<std::vector<double>> &testX,
    std::vector<std::vector<double>> &testY,
    size_t trainSize = 60000,
    size_t testSize = 10000,
    size_t trainOffset = 0,
    size_t testOffset = 0
);
