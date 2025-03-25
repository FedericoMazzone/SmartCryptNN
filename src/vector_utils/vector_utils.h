#pragma once

#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>


/**
 * Convert a vector to a string representation.
 *
 * @param vector The vector to convert.
 * @return A string representation of the vector.
 *
 * @example
 *     std::vector<int> v = {1, 2, 3};
 *     std::string s = vec2str(v);
 *     // s = "[ 1 2 3 ]"
 *
 * @example
 *     std::vector<std::string> v = {"foo", "bar", "baz"};
 *     std::string s = vec2str(v);
 *     // s = "[ foo bar baz ]"
 */
template<typename T>
std::string vec2str(
    const std::vector<T> &vector
)
{
    std::ostringstream oss;
    oss << "[ ";
    for (T v : vector)
        oss << v << " ";
    oss << "]" << std::endl;

    return oss.str();
}


/**
 * Convert a matrix to a string representation.
 *
 * @param matrix The matrix to convert.
 * @return A string representation of the matrix.
 */
template<typename T>
std::string mat2str(
    const std::vector<std::vector<T>> &matrix
)
{
    std::ostringstream oss;
    oss << "[ " << std::endl;
    for (std::vector<T> vector : matrix)
    {
        oss << "   [ ";
        for (T v : vector)
            oss << v << " ";
        oss << "]" << std::endl;
    }
    oss << "]" << std::endl;

    return oss.str();
}


/**
 * Print vector.
 * @param vector vector to print
 */
void printVector(
    const std::vector<int64_t> &vector
);


/**
 * Print vector.
 * @param vector vector to print
 */
void printVector(
    const std::vector<double> &vector
);


/**
 * Print matrix.
 * @param matrix matrix to print
 */
void printMatrix(
    const std::vector<std::vector<int64_t>> &matrix
);


/**
 * Print matrix.
 * @param matrix matrix to print
 */
void printMatrix(
    const std::vector<std::vector<double>> &matrix
);


/**
 * Add two vectors elementwise.
 * @param vector1 first input vector
 * @param vector2 second input vector
 * @param modulus optional modulus
 * @return the sum of the two input vectors
 */
std::vector<int64_t> addVectors(
    const std::vector<int64_t> &vector1,
    const std::vector<int64_t> &vector2,
    const int64_t &modulus = 0
);


/**
 * Add two vectors elementwise.
 * @param vector1 first input vector
 * @param vector2 second input vector
 * @return the sum of the two input vectors
 */
std::vector<double> addVectors(
    const std::vector<double> &vector1,
    const std::vector<double> &vector2
);


/**
 * Add two matrices elementwise.
 * @param vector1 first input matrix
 * @param vector2 second input matrix
 * @return the sum of the two input matrices
 */
std::vector<std::vector<double>> addMatrices(
    const std::vector<std::vector<double>> &matrix1,
    const std::vector<std::vector<double>> &matrix2
);


/**
 * Multiply two vectors elementwise.
 * @param vector1 first input vector
 * @param vector2 second input vector
 * @return the product of the two input vectors
 */
std::vector<int64_t> multVectors(
    const std::vector<int64_t> &vector1,
    const std::vector<int64_t> &vector2
);


/**
 * Multiply two vectors elementwise.
 * @param vector1 first input vector
 * @param vector2 second input vector
 * @return the product of the two input vectors
 */
std::vector<double> multVectors(
    const std::vector<double> &vector1,
    const std::vector<double> &vector2
);


/**
 * Multiply vector times scalar.
 * @param scalar input scalar
 * @param vector input vector
 * @return the product of the vector times the scalar
 */
std::vector<double> multVectors(
    const double scalar,
    const std::vector<double> &vector
);


/**
 * Generate random valuein [-maxValue, maxValue).
 * @param maxValue absolute maximum value
 * @return random value
 */
int64_t getRandValue(
    int64_t maxValue
);


/**
 * Generate random valuein [-maxValue, maxValue).
 * @param maxValue absolute maximum value
 * @return random value
 */
double getRandValue(
    double maxValue
);


/**
 * Generate random vector of size length, with values in [-maxValue, maxValue).
 * @param length desired length of the vector
 * @param maxValue absolute maximum value of the coefficients
 * @return random vector
 */
std::vector<int64_t> genRandVect(
    size_t length,
    int64_t maxValue
);


/**
 * Generate random vector of size length, with values in [-maxValue, maxValue).
 * @param length desired length of the vector
 * @param maxValue absolute maximum value of the coefficients
 * @return random vector
 */
std::vector<double> genRandVect(
    size_t length,
    double maxValue
);


/**
 * Generate random vector of size length, with values picked from normal
 * distribution with given mean and standard deviation.
 * @param length desired length of the vector
 * @param mean mean of the normal distribution
 * @param stddev standard deviation of the normal distribution
 * @return random vector
 */
std::vector<double> genRandNormalVector(
    size_t length,
    double mean,
    double stddev
);


/**
 * Generate random matrix of size (rows x cols), with values in [-maxValue,
 * maxValue).
 * @param rows desired number of rows
 * @param cols desired number of columns
 * @param maxValue absolute maximum value of the coefficients
 * @return random matrix
 */
std::vector<std::vector<int64_t>> genRandMatrix(
    size_t rows,
    size_t cols,
    int64_t maxValue
);


/**
 * Generate random matrix of size (rows x cols), with values in [-maxValue,
 * maxValue).
 * @param rows desired number of rows
 * @param cols desired number of columns
 * @param maxValue absolute maximum value of the coefficients
 * @return random matrix
 */
std::vector<std::vector<double>> genRandMatrix(
    size_t rows,
    size_t cols,
    double maxValue
);


/**
 * Generate random matrix of size (rows x cols), with values picked from normal
 * distribution with given mean and standard deviation.
 * @param rows desired number of rows
 * @param cols desired number of columns
 * @param mean mean of the normal distribution
 * @param stddev standard deviation of the normal distribution
 * @return random matrix
 */
std::vector<std::vector<double>> genRandNormalMatrix(
    size_t rows,
    size_t cols,
    double mean,
    double stddev
);


/**
 * Transpose the given matrix.
 * @param matrix input matrix
 * @return transposed matrix
 */
std::vector<std::vector<int64_t>> transpose(
    std::vector<std::vector<int64_t>> matrix
);


/**
 * Transpose the given matrix.
 * @param matrix input matrix
 * @return transposed matrix
 */
std::vector<std::vector<double>> transpose(
    std::vector<std::vector<double>> matrix
);


/**
 * Compute the least power of two greater or equal than the input value.
 * @param n input value
 * @return least power of two >= n
 */
size_t nextPowerOf2(
    size_t n
);


/**
 * Resize the input matrix to reach the desired number of rows and columns, by
 * padding with 0s if necessary.
 * @param matrix input matrix
 * @param numRows output's number of rows
 * @param numCols output's number of columns
 * @return resized matrix
 */
std::vector<std::vector<int64_t>> resizeMatrix(
    std::vector<std::vector<int64_t>> matrix,
    size_t numRows,
    size_t numCols
);


/**
 * Resize the input matrix to reach the desired number of rows and columns, by
 * padding with 0s if necessary.
 * @param matrix input matrix
 * @param numRows output's number of rows
 * @param numCols output's number of columns
 * @return resized matrix
 */
std::vector<std::vector<double>> resizeMatrix(
    std::vector<std::vector<double>> matrix,
    size_t numRows,
    size_t numCols
);


/**
 * Flatten the input matrix.
 * @param matrix input matrix
 * @param direction true for row-wise, false for column-wise
 * @return flattened matrix
 */
std::vector<int64_t> flattenMatrix(
    std::vector<std::vector<int64_t>> matrix,
    bool direction = true
);


/**
 * Flatten the input matrix.
 * @param matrix input matrix
 * @param direction true for row-wise, false for column-wise
 * @return flattened matrix
 */
std::vector<double> flattenMatrix(
    std::vector<std::vector<double>> matrix,
    bool direction = true
);


/**
 * Scale up a double vector to int for a given scale.
 * @param vector input vector
 * @param scale scale
 * @return scaled vector (scale * vector)
 */
std::vector<int64_t> scaleVector(
    const std::vector<double> &vector,
    const int scale
);


/**
 * Scale up a double matrix to int for a given scale.
 * @param matrix input matrix
 * @param scale scale
 * @return scaled matrix (scale * matrix)
 */
std::vector<std::vector<int64_t>> scaleMatrix(
    const std::vector<std::vector<double>> &matrix,
    const int scale
);


/**
 * Compute the argmax of a given vector.
 * @param vector input vector
 * @return argmax(vector)
 */
size_t argmax(
    const std::vector<int64_t> &vector
);


/**
 * Compute the argmax of a given vector.
 * @param vector input vector
 * @return argmax(vector)
 */
size_t argmax(
    const std::vector<double> &vector
);


/**
 * Compute the max absolute value of a given vector.
 * @param vector input vector
 * @return max(abs(vector))
 */
double maxAbs(
    const std::vector<double> &vector
);


/**
 * Compute the max absolute value of a given matrix.
 * @param matrix input matrix
 * @return max(abs(matrix))
 */
double maxAbs(
    const std::vector<std::vector<double>> &matrix
);


/**
 * Compute the mod operation in the range [- mod / 2, ... mod / 2).
 * @param value input value
 * @param modulus input modulus
 * @return representative of value (mod modulus)
 */
int64_t mod(
    int64_t value,
    const int64_t &modulus
);


/**
 * Compute vector-matrix multiplication.
 * @param vector input vector
 * @param matrix input matrix
 * @return vector-matrix product
 */
std::vector<int64_t> vectorMatrixMult(
    std::vector<int64_t> vector,
    std::vector<std::vector<int64_t>> matrix
);


/**
 * Compute vector-matrix multiplication.
 * @param vector input vector
 * @param matrix input matrix
 * @return vector-matrix product
 */
std::vector<double> vectorMatrixMult(
    std::vector<double> vector,
    std::vector<std::vector<double>> matrix
);


/**
 * Compute matrix-matrix multiplication.
 * @param matrix1 the first matrix
 * @param matrix2 the second matrix
 * @return matrix-matrix product
 */
std::vector<std::vector<int64_t>> matrixMult(
    std::vector<std::vector<int64_t>> matrix1,
    std::vector<std::vector<int64_t>> matrix2
);


/**
 * Compute matrix-matrix multiplication.
 * @param matrix1 the first matrix
 * @param matrix2 the second matrix
 * @return matrix-matrix product
 */
std::vector<std::vector<double>> matrixMult(
    std::vector<std::vector<double>> matrix1,
    std::vector<std::vector<double>> matrix2
);


/**
 * Multiply matrix by a scalar.
 * @param scalar the scalar
 * @param matrix the matrix
 * @return scalar-matrix product
 */
std::vector<std::vector<double>> matrixMult(
    const double scalar,
    std::vector<std::vector<double>> matrix
);


/**
 * Apply sigmoid function to a vector.
 * @param input the input vector
 * @return sigmoid(input) applied element-wise
 */
std::vector<double> sigmoid(
    std::vector<double> input
);


/**
 * Apply the derivative of the sigmoid function to a vector.
 * @param input the input vector
 * @return sigmoid'(input) applied element-wise
 */
std::vector<double> sigmoidDerivative(
    std::vector<double> input
);


std::vector<double> operator+(
    const std::vector<double> &a,
    const std::vector<double> &b
);


std::vector<double> operator-(
    const std::vector<double> &a,
    const std::vector<double> &b
);


std::vector<double> operator*(
    const double &a,
    const std::vector<double> &b
);


std::vector<double> operator*(
    const std::vector<double> &a,
    const double &b
);


std::vector<double> operator*(
    const std::vector<double> &a,
    const std::vector<double> &b
);


std::vector<std::vector<double>> operator+(
    const std::vector<std::vector<double>> &a,
    const std::vector<std::vector<double>> &b
);


std::vector<std::vector<double>> operator-(
    const std::vector<std::vector<double>> &a,
    const std::vector<std::vector<double>> &b
);


std::vector<std::vector<double>> operator*(
    const double &a,
    const std::vector<std::vector<double>> &b
);


std::vector<std::vector<double>> operator*(
    const std::vector<std::vector<double>> &b,
    const double &a
);


std::vector<double> operator*(
    const std::vector<double> &a,
    const std::vector<std::vector<double>> &b
);


std::vector<double> operator*(
    const std::vector<std::vector<double>> &a,
    const std::vector<double> &b
);


std::vector<std::vector<double>> operator*(
    const std::vector<std::vector<double>> &a,
    const std::vector<std::vector<double>> &b
);


std::string double2str(
    const double &value
);


std::string vec2str(
    const std::vector<double> &vector
);


std::vector<double> str2vec(
    const std::string &s
);


std::string vec2str(
    const std::vector<size_t> &vector
);


std::vector<size_t> str2intvec(
    const std::string &s
);


std::string mtx2str(
    const std::vector<std::vector<double>> &matrix
);


std::vector<std::vector<double>> str2mtx(
    const std::string &s
);
