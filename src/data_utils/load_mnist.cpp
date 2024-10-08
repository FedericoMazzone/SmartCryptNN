#include "load_mnist.h"

#include <iostream>
#include <fstream>
#include <string>



void loadMNIST(
    std::vector<std::vector<int64_t>> &trainX,
    std::vector<int64_t> &trainY,
    std::vector<std::vector<int64_t>> &testX,
    std::vector<int64_t> &testY
)
{
    std::string line;
    std::ifstream inputStream;
    std::vector<int64_t> sample;
    size_t id, i;

    inputStream.open("data/mnist.txt");
    if (!inputStream.is_open())
        std::cerr << "Error opening file" << std::endl;

    id = 0;
    while (std::getline(inputStream, line)) {
        for (i = 0; i < line.length() - 1; i++)
            sample.push_back(line[i] - '0');
        if (id < 60000) {
            trainX.push_back(sample);
            trainY.push_back(line[i] - '0');
        } else {
            testX.push_back(sample);
            testY.push_back(line[i] - '0');
        }
        sample.clear();
        id++;
    }

    inputStream.close();
}


void loadMNIST(
    std::vector<std::vector<double>> &trainX,
    std::vector<std::vector<double>> &trainY,
    std::vector<std::vector<double>> &testX,
    std::vector<std::vector<double>> &testY,
    size_t trainSize,
    size_t testSize,
    size_t trainOffset,
    size_t testOffset
)
{
    std::vector<std::vector<int64_t>> trainXint, testXint;
    std::vector<int64_t> trainYint, testYint;
    loadMNIST(trainXint, trainYint, testXint, testYint);

    for (size_t i = trainOffset; i < trainOffset + trainSize; i++)
    {
        std::vector<int64_t> xint = trainXint[i];
        std::vector<double> x(xint.begin(), xint.end());
        trainX.push_back(x);

        std::vector<double> y(10);
        y[trainYint[i]] = 1.0;
        trainY.push_back(y);
    }

    for (size_t i = testOffset; i < testOffset + testSize; i++)
    {
        std::vector<int64_t> xint = testXint[i];
        std::vector<double> x(xint.begin(), xint.end());
        testX.push_back(x);

        std::vector<double> y(10);
        y[testYint[i]] = 1.0;
        testY.push_back(y);
    }
}


// int main(int argc, char* argv[]) {
    
//     std::vector<std::vector<int64_t>> trainX, testX;
//     std::vector<int64_t> trainY, testY;

//     loadMNIST(trainX, trainY, testX, testY);

//     for (size_t i = 0; i < trainX.size(); i++) {
//         std::cout << i << " ";
//         for (auto value : trainX[i])
//             std::cout << value;
//         std::cout << trainY[i] << std::endl;
//     }

//     for (size_t i = 0; i < testX.size(); i++) {
//         std::cout << i << " ";
//         for (auto value : testX[i])
//             std::cout << value;
//         std::cout << testY[i] << std::endl;
//     }

// }
