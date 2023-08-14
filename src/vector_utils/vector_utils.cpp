#include "vector_utils.h"

#include <algorithm>
#include <cmath>
#include <ctime>
#include <random>


void printVector(
    const std::vector<int64_t> &vector
)
{
    std::cout << "[ ";
    for (int64_t v : vector)
        std::cout << v << " ";
    std::cout << "]" << std::endl;
}


void printVector(
    const std::vector<double> &vector
)
{
    std::cout << "[ ";
    for (double v : vector)
        std::cout << v << " ";
    std::cout << "]" << std::endl;
}


void printMatrix(
    const std::vector<std::vector<int64_t>> &matrix
)
{
    std::cout << "[" << std::endl;
    for (std::vector<int64_t> vector : matrix) {
        std::cout << "   [ ";
        for (int64_t v : vector)
            std::cout << v << " ";
        std::cout << "]" << std::endl;
    }
    std::cout << "]" << std::endl;
}


void printMatrix(
    const std::vector<std::vector<double>> &matrix
)
{
    std::cout << "[" << std::endl;
    for (std::vector<double> vector : matrix) {
        std::cout << "   [ ";
        for (double v : vector)
            std::cout << v << " ";
        std::cout << "]" << std::endl;
    }
    std::cout << "]" << std::endl;
}


std::vector<int64_t> addVectors(
    const std::vector<int64_t> &vector1,
    const std::vector<int64_t> &vector2,
    const int64_t &modulus
)
{
    std::vector<int64_t> result;
    if (modulus == 0)
        for (size_t i = 0; i < vector1.size(); i++)
            result.push_back(vector1[i] + vector2[i]);
    else
        for (size_t i = 0; i < vector1.size(); i++)
            result.push_back(mod(vector1[i] + vector2[i], modulus));
    return result;
}


std::vector<double> addVectors(
    const std::vector<double> &vector1,
    const std::vector<double> &vector2
)
{
    std::vector<double> result;
    for (size_t i = 0; i < vector1.size(); i++)
        result.push_back(vector1[i] + vector2[i]);
    return result;
}


std::vector<std::vector<double>> addMatrices(
    const std::vector<std::vector<double>> &matrix1,
    const std::vector<std::vector<double>> &matrix2
)
{
    std::vector<std::vector<double>> result;
    for (size_t i = 0; i < matrix1.size(); i++)
        result.push_back(addVectors(matrix1[i], matrix2[i]));
    return result;
}


std::vector<int64_t> multVectors(
    const std::vector<int64_t> &vector1,
    const std::vector<int64_t> &vector2
)
{
    std::vector<int64_t> result;
    for (size_t i = 0; i < vector1.size(); i++)
        result.push_back(vector1[i] * vector2[i]);
    return result;
}


std::vector<double> multVectors(
    const std::vector<double> &vector1,
    const std::vector<double> &vector2
)
{
    std::vector<double> result;
    for (size_t i = 0; i < vector1.size(); i++)
        result.push_back(vector1[i] * vector2[i]);
    return result;
}


std::vector<double> multVectors(
    const double scalar,
    const std::vector<double> &vector
)
{
    std::vector<double> result;
    for (size_t i = 0; i < vector.size(); i++)
        result.push_back(vector[i] * scalar);
    return result;
}


int64_t getRandValue(
    int64_t maxValue
)
{
    return (std::rand() % (maxValue << 1)) - maxValue;
}


double getRandValue(
    double maxValue
)
{
    return static_cast<double>(rand()) / (static_cast<double>(RAND_MAX / (maxValue * 2))) - maxValue;
}


std::vector<int64_t> genRandVect(
    size_t length,
    int64_t maxValue
)
{
    auto myrand = [maxValue] () {return getRandValue(maxValue);};
    std::vector<int64_t> vector(length);
    std::generate(vector.begin(), vector.end(), myrand);
    return vector;
}


std::vector<double> genRandVect(
    size_t length,
    double maxValue
)
{
    auto myrand = [maxValue] () {return getRandValue(maxValue);};
    std::vector<double> vector(length);
    std::generate(vector.begin(), vector.end(), myrand);
    return vector;
}


std::vector<double> genRandNormalVector(
    size_t length,
    double mean,
    double stddev
)
{
    static std::default_random_engine generator;
    std::normal_distribution<double> distribution(mean, stddev);
    std::vector<double> vector(length);
    for (size_t i = 0; i < length; i++)
        vector[i] = distribution(generator);
    return vector;
}


std::vector<std::vector<int64_t>> genRandMatrix(
    size_t rows,
    size_t cols,
    int64_t maxValue
)
{
    auto myrand = [maxValue] () {return getRandValue(maxValue);};
    std::vector<std::vector<int64_t>> matrix(rows, std::vector<int64_t>(cols));
    for (size_t i = 0; i < rows; i++)
        std::generate(matrix[i].begin(), matrix[i].end(), myrand);
    return matrix;
}


std::vector<std::vector<double>> genRandMatrix(
    size_t rows,
    size_t cols,
    double maxValue
)
{
    auto myrand = [maxValue] () {return getRandValue(maxValue);};
    std::vector<std::vector<double>> matrix(rows, std::vector<double>(cols));
    for (size_t i = 0; i < rows; i++)
        std::generate(matrix[i].begin(), matrix[i].end(), myrand);
    return matrix;
}


std::vector<std::vector<double>> genRandNormalMatrix(
    size_t rows,
    size_t cols,
    double mean,
    double stddev
)
{
    static std::default_random_engine generator(5);
    std::normal_distribution<double> distribution(mean, stddev);
    std::vector<std::vector<double>> matrix(rows, std::vector<double>(cols));
    for (size_t i = 0; i < rows; i++)
        for (size_t j = 0; j < cols; j++)
            matrix[i][j] = distribution(generator);
    return matrix;
}


std::vector<std::vector<int64_t>> transpose(
    std::vector<std::vector<int64_t>> matrix
)
{
    std::vector<std::vector<int64_t>> matrixT(
        matrix[0].size(),
        std::vector<int64_t>(matrix.size())
    );
    for (size_t i = 0; i < matrix[0].size(); i++) 
        for (size_t j = 0; j < matrix.size(); j++)
            matrixT[i][j] = matrix[j][i];
    return matrixT;
}


std::vector<std::vector<double>> transpose(
    std::vector<std::vector<double>> matrix
)
{
    std::vector<std::vector<double>> matrixT(
        matrix[0].size(),
        std::vector<double>(matrix.size())
    );
    for (size_t i = 0; i < matrix[0].size(); i++) 
        for (size_t j = 0; j < matrix.size(); j++)
            matrixT[i][j] = matrix[j][i];
    return matrixT;
}


size_t nextPowerOf2(
    size_t n
)
{
    if (n == 0 || n == 1) return 1;
    else return 1 << ((int) std::log2(n - 1) + 1);
}


std::vector<std::vector<int64_t>> resizeMatrix(
    std::vector<std::vector<int64_t>> matrix,
    size_t numRows,
    size_t numCols
)
{
    for (auto &row : matrix) row.resize(numCols, 0);
    matrix.resize(numRows, std::vector<int64_t>(numCols, 0));
    return matrix;
}


std::vector<std::vector<double>> resizeMatrix(
    std::vector<std::vector<double>> matrix,
    size_t numRows,
    size_t numCols
)
{
    for (auto &row : matrix) row.resize(numCols, 0);
    matrix.resize(numRows, std::vector<double>(numCols, 0));
    return matrix;
}


std::vector<int64_t> flattenMatrix(
    std::vector<std::vector<int64_t>> matrix,
    bool direction
)
{
    std::vector<int64_t> res;
    if (direction)
        for (auto &row : matrix)
            res.insert(end(res), begin(row), end(row));
    else {
        for (size_t i = 0; i < matrix[0].size(); i++) 
            for (size_t j = 0; j < matrix.size(); j++)
                res.push_back(matrix[j][i]);
    }
    return res;
}


std::vector<double> flattenMatrix(
    std::vector<std::vector<double>> matrix,
    bool direction
)
{
    std::vector<double> res;
    if (direction)
        for (auto &row : matrix)
            res.insert(end(res), begin(row), end(row));
    else {
        for (size_t i = 0; i < matrix[0].size(); i++) 
            for (size_t j = 0; j < matrix.size(); j++)
                res.push_back(matrix[j][i]);
    }
    return res;
}


std::vector<int64_t> scaleVector(
    const std::vector<double> &vector,
    const int scale
)
{
    std::vector<int64_t> result;
    for (double v : vector)
        result.push_back(v * scale);
    return result;
}


std::vector<std::vector<int64_t>> scaleMatrix(
    const std::vector<std::vector<double>> &matrix,
    const int scale
)
{
    std::vector<std::vector<int64_t>> result;
    std::vector<int64_t> row;
    for (std::vector<double> vector : matrix) {
        for (double v : vector)
            row.push_back(v * scale);
        result.push_back(row);
        row.clear();
    }
    return result;
}


size_t argmax(
    const std::vector<int64_t> &vector
)
{
    size_t argmax = 0;
    int64_t max = vector[argmax];
    for (size_t i = 0; i < vector.size(); i++) {
        if (vector[i] > max) {
            argmax = i;
            max = vector[i];
        }
    }
    return argmax;
}


size_t argmax(
    const std::vector<double> &vector
)
{
    size_t argmax = 0;
    double max = vector[argmax];
    for (size_t i = 0; i < vector.size(); i++) {
        if (vector[i] > max) {
            argmax = i;
            max = vector[i];
        }
    }
    return argmax;
}


double maxAbs(
    const std::vector<double> &vector
)
{
    double max = 0.0;
    double candidate;
    for (size_t i = 0; i < vector.size(); i++)
    {
        candidate = vector[i];
        if (candidate > max)
            max = candidate;
        else if (candidate < -max)
            max = -candidate;
    }
    return max;
}


double maxAbs(
    const std::vector<std::vector<double>> &matrix
)
{
    double max = 0.0;
    double candidate;
    for (const std::vector<double> &row : matrix)
    {
        candidate = maxAbs(row);
        if (candidate > max)
            max = candidate;
    }
    return max;
}


int64_t mod(
    int64_t value,
    const int64_t &modulus
)
{
    value = value % modulus;

    if (value > ((modulus % 2 == 0) ? (modulus >> 1) - 1 : (modulus >> 1)))
        value -= modulus;
    else if (value < - (modulus >> 1))
        value += modulus;

    return value;
}



/**
 * Compute the inner product between two vectors.
 * @param vector1 first input vector
 * @param vector2 second input vector
 * @return inner product value
 */
int64_t innerProduct(
    std::vector<int64_t> vector1,
    std::vector<int64_t> vector2
)
{
    int64_t inner_product = 0;
    for (size_t i = 0; i < vector1.size(); i++)
        inner_product += vector1[i] * vector2[i];
    return inner_product;
}


/**
 * Compute the inner product between two vectors.
 * @param vector1 first input vector
 * @param vector2 second input vector
 * @return inner product value
 */
double innerProduct(
    std::vector<double> vector1,
    std::vector<double> vector2
)
{
    double inner_product = 0.0;
    for (size_t i = 0; i < vector1.size(); i++)
        inner_product += vector1[i] * vector2[i];
    return inner_product;
}


std::vector<int64_t> vectorMatrixMult(
    std::vector<int64_t> vector,
    std::vector<std::vector<int64_t>> matrix
)
{
    std::vector<std::vector<int64_t>> matrixT = transpose(matrix);
    std::vector<int64_t> result;
    for (size_t i = 0; i < matrixT.size(); i++) {
        int64_t innProd = innerProduct(vector, matrixT[i]);
        result.push_back(innProd);
    }
    return result;
}


std::vector<double> vectorMatrixMult(
    std::vector<double> vector,
    std::vector<std::vector<double>> matrix
)
{
    std::vector<std::vector<double>> matrixT = transpose(matrix);
    std::vector<double> result;
    for (size_t i = 0; i < matrixT.size(); i++) {
        double innProd = innerProduct(vector, matrixT[i]);
        result.push_back(innProd);
    }
    return result;
}


std::vector<std::vector<int64_t>> matrixMult(
    std::vector<std::vector<int64_t>> matrix1,
    std::vector<std::vector<int64_t>> matrix2
)
{
    std::vector<std::vector<int64_t>> result;
    for (std::vector<int64_t> row : matrix1)
        result.push_back(vectorMatrixMult(row, matrix2));
    return result;
}


std::vector<std::vector<double>> matrixMult(
    std::vector<std::vector<double>> matrix1,
    std::vector<std::vector<double>> matrix2
)
{
    std::vector<std::vector<double>> result;
    for (std::vector<double> row : matrix1)
        result.push_back(vectorMatrixMult(row, matrix2));
    return result;
}


std::vector<std::vector<double>> matrixMult(
    const double scalar,
    std::vector<std::vector<double>> matrix
)
{
    for (size_t i = 0; i < matrix.size(); i++)
        for (size_t j = 0; j < matrix[0].size(); j++)
            matrix[i][j] *= scalar;
    return matrix;
}


std::vector<double> sigmoid(
    std::vector<double> input
)
{
    for (double &v : input)
        v = 1 / (1 + exp(-v));
    return input;
}


std::vector<double> sigmoidDerivative(
    std::vector<double> input
)
{
    for (double &v : input)
        v = exp(-v) / ((1 + exp(-v)) * (1 + exp(-v)));
    return input;
}


std::vector<double> operator+(
    const std::vector<double> &a,
    const std::vector<double> &b
)
{
    return addVectors(a, b);
}


std::vector<double> operator-(
    const std::vector<double> &a,
    const std::vector<double> &b
)
{
    return a + multVectors(-1, b);
}


std::vector<double> operator*(
    const double &a,
    const std::vector<double> &b
)
{
    return multVectors(a, b);
}


std::vector<double> operator*(
    const std::vector<double> &a,
    const double &b
)
{
    return b * a;
}


std::vector<double> operator*(
    const std::vector<double> &a,
    const std::vector<double> &b
)
{
    return multVectors(a, b);
}


std::vector<std::vector<double>> operator+(
    const std::vector<std::vector<double>> &a,
    const std::vector<std::vector<double>> &b
)
{
    return addMatrices(a, b);
}


std::vector<std::vector<double>> operator-(
    const std::vector<std::vector<double>> &a,
    const std::vector<std::vector<double>> &b
)
{
    return a + matrixMult(-1, b);
}


std::vector<std::vector<double>> operator*(
    const double &a,
    const std::vector<std::vector<double>> &b
)
{
    return matrixMult(a, b);
}


std::vector<std::vector<double>> operator*(
    const std::vector<std::vector<double>> &a,
    const double &b
)
{
    return b * a;
}


std::vector<double> operator*(
    const std::vector<double> &a,
    const std::vector<std::vector<double>> &b
)
{
    return vectorMatrixMult(a, b);
}


std::vector<double> operator*(
    const std::vector<std::vector<double>> &a,
    const std::vector<double> &b
)
{
    return vectorMatrixMult(b, transpose(a));
}


std::vector<std::vector<double>> operator*(
    const std::vector<std::vector<double>> &a,
    const std::vector<std::vector<double>> &b
)
{
    return matrixMult(a, b);
}


std::string double2str(
    const double &value
)
{
    char s[22];
    int ret = snprintf(s, sizeof(s), "%21.21f", value);
    if (ret < 0) abort();
    return s;
}


std::string vec2str(
    const std::vector<double> &vector
)
{
    std::string s("[ ");

    for (size_t i = 0; i < vector.size(); i++)
    {
        s += double2str(vector[i]);
        if (i < vector.size() - 1)
            s += ", ";
    }
    s += " ]";

    return s;
}


std::vector<double> str2vec(
    const std::string &s
)
{
    size_t pos1 = s.find('[') + 1;
    size_t pos2 = s.find(']');
    std::string t = s.substr(pos1, pos2 - pos1);

    std::vector<double> result;
    std::stringstream ss(t);
    while (ss.good())
    {
        std::string substr;
        getline(ss, substr, ',');
        result.push_back(std::stod(substr));
    }

    return result;
}


std::string vec2str(
    const std::vector<size_t> &vector
)
{
    std::string s("[ ");

    for (size_t i = 0; i < vector.size(); i++)
    {
        s += std::to_string(vector[i]);
        if (i < vector.size() - 1)
            s += ", ";
    }
    s += " ]";

    return s;
}


std::vector<size_t> str2intvec(
    const std::string &s
)
{
    size_t pos1 = s.find('[') + 1;
    size_t pos2 = s.find(']');
    std::string t = s.substr(pos1, pos2 - pos1);

    std::vector<size_t> result;
    std::stringstream ss(t);
    while (ss.good())
    {
        std::string substr;
        getline(ss, substr, ',');
        result.push_back(std::stoi(substr));
    }

    return result;
}


std::string mtx2str(
    const std::vector<std::vector<double>> &matrix
)
{
    std::string s("[ ");

    for (size_t i = 0; i < matrix.size(); i++)
    {
        s += vec2str(matrix[i]);
        if (i < matrix.size() - 1)
            s += ", ";
    }
    s += " ]";

    return s;
}


std::vector<std::vector<double>> str2mtx(
    const std::string &s
)
{
    std::vector<std::vector<double>> result;

    std::string t = s.substr(s.find('[') + 1);
    size_t pos1;
    while ((pos1 = t.find('[')) != std::string::npos)
    {
        result.push_back(str2vec(t));
        t = t.substr(t.find(']') + 1);
    }

    return result;
}
