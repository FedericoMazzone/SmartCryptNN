//==================================================================================
// BSD 2-Clause License
//
// Copyright (c) 2014-2022, NJIT, Duality Technologies Inc. and other contributors
//
// All rights reserved.
//
// Author TPOC: contact@openfhe.org
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//==================================================================================


#include "openfhe.h"

using namespace lbcrypto;


/**
 * Pad and flatten a matrix.
 * @param matrix input matrix
 * @param numRows variable to store the number of rows of the padded matrix
 * @param numCols variable to store the number of columns of the padded matrix
 * @param packing true for column-wise, false for row-wise
 * @param numRowsPrevMatrix only needed for row-wise packing
 * @return encoded matrix
 */
std::vector<double> encodeMatrix(
    std::vector<std::vector<double>> matrix,
    size_t &numRows,
    size_t &numCols,
    const bool packing,
    const size_t numRowsPrevMatrix
);


/**
 * Pad and flatten a matrix.
 * @param cryptoContext the crypto context
 * @param matrix (plaintext) input matrix
 * @param numRows variable to store the number of rows of the padded matrix
 * @param numCols variable to store the number of columns of the padded matrix
 * @param packing true for column-wise, false for row-wise
 * @param numRowsPrevMatrix only needed for row-wise packing
 * @return encoded matrix
 */
Plaintext encodeMatrix(
    CryptoContext<DCRTPoly> cryptoContext,
    std::vector<std::vector<double>> matrix,
    size_t &numRows,
    size_t &numCols,
    const bool packing = true,
    const size_t numRowsPrevMatrix = 0
);


/**
 * Decode a plaintext matrix from the column- or row-based packing.
 * @param matrixP (plaintext) input matrix
 * @param ogNumRows original number of rows of the matrix
 * @param ogNumCols original number of columns of the matrix
 * @param packing true for column-wise, false for row-wise
 * @param numRowsPrevMatrix only needed for row-wise packing
 * @return decoded matrix
 */
std::vector<std::vector<double>> decodeMatrix(
    const Plaintext &matrixP,
    const size_t ogNumRows,
    const size_t ogNumCols,
    const bool packing = true,
    const size_t numRowsPrevMatrix = 0
);


std::vector<double> decodeVector(
    const Plaintext &vectorP,
    const size_t ogLength,
    const bool packing,
    const size_t numRowsPrevMatrix = 0
);


/**
 * Mask an encrypted vector.
 * @param cryptoContext the crypto context
 * @param ciphertext the encrypted vector
 * @param packing true for column-wise, false for row-wise
 * @param ogNumCols
 * @param numRows
 * @param numCols
 * @return masked vector
 */
Ciphertext<DCRTPoly> maskVector(
    CryptoContext<DCRTPoly> cryptoContext,
    Ciphertext<DCRTPoly> ciphertext,
    bool packing,
    size_t ogNumCols,
    size_t numRows = 0,
    size_t numCols = 0
);


/**
 * Compute the product between an encrypted vector and a plaintext matrix.
 * Matrix packing approach from Kim et al. Logistic regression model training
 * based on the approximate homomorphic encryption. 
 * Alternate packing approach from Sav et al. Poseidon: Privacy-preserving
 * federated neural network learning.
 * @param cryptoContext the crypto context
 * @param publicKey the public key
 * @param vectorC encrypted input vector
 * @param matrix (plaintext) input matrix
 * @param packing true for column-wise, false for row-wise
 * @param numRowsPrevMatrix only needed for row-wise packing
 * @param masking whether you want the output ciphertext to contain only the
 * output value in the first positions, and 0s in the other positions
 * @param transposing only for column-wise packing, whether you want the output
 * vector to be transposed (in terms of packing) and so be ready for a new
 * column-wise pack multiplication
 * @return encrypted vector-matrix product
 */
Ciphertext<DCRTPoly> vectorMatrixMultPackCP(
    CryptoContext<DCRTPoly> cryptoContext,
    PublicKey<DCRTPoly> publicKey,
    Ciphertext<DCRTPoly> vectorC,
    std::vector<std::vector<int64_t>> matrix,
    bool packing = true,
    int numRowsPrevMatrix = -1,
    bool masking = true,
    bool transposing = true
);


/**
 * Compute the product between an encrypted vector and a plaintext matrix.
 * Matrix packing approach from Kim et al. Logistic regression model training
 * based on the approximate homomorphic encryption. 
 * Alternate packing approach from Sav et al. Poseidon: Privacy-preserving
 * federated neural network learning.
 * @param cryptoContext the crypto context
 * @param publicKey the public key
 * @param vectorC encrypted input vector
 * @param matrix (plaintext) input matrix
 * @param packing true for column-wise, false for row-wise
 * @param numRowsPrevMatrix only needed for row-wise packing
 * @param masking whether you want the output ciphertext to contain only the
 * output value in the first positions, and 0s in the other positions
 * @param transposing only for column-wise packing, whether you want the output
 * vector to be transposed (in terms of packing) and so be ready for a new
 * column-wise pack multiplication
 * @return encrypted vector-matrix product
 */
Ciphertext<DCRTPoly> vectorMatrixMultPackCP(
    CryptoContext<DCRTPoly> cryptoContext,
    PublicKey<DCRTPoly> publicKey,
    Ciphertext<DCRTPoly> vectorC,
    std::vector<std::vector<double>> matrix,
    bool packing = true,
    int numRowsPrevMatrix = -1,
    bool masking = true,
    bool transposing = true
);


/**
 * Compute the product between an encrypted vector and an encrypted matrix.
 * @param cryptoContext the crypto context
 * @param publicKey the public key
 * @param vectorC encrypted input vector
 * @param matrixC encrypted input matrix
 * @param ogNumCols the original number of columns of the matrix
 * @param numRows the number of rows of the matrix
 * @param numCols the number of columns of the matrix
 * @param packing true for column-wise, false for row-wise
 * @param masking whether you want the output ciphertext to contain only the
 * output value in the first positions, and 0s in the other positions
 * @param transposing only for column-wise packing, whether you want the output
 * vector to be transposed (in terms of packing) and so be ready for a new
 * column-wise pack multiplication
 * @return encrypted vector-matrix product
 */
Ciphertext<DCRTPoly> vectorMatrixMultPackCC(
    CryptoContext<DCRTPoly> cryptoContext,
    PublicKey<DCRTPoly> publicKey,
    Ciphertext<DCRTPoly> vectorC,
    Ciphertext<DCRTPoly> matrixC,
    size_t ogNumCols,
    size_t numRows,
    size_t numCols,
    bool packing = true,
    bool masking = true,
    bool transposing = true
);


/**
 * Compute the matrix product transpose(v1) * v2.
 * @param cryptoContext the crypto context
 * @param vector1C the first encrypted input vector
 * @param vector2C the second encrypted input vector
 * @param length1 the length of the first input vector
 * @param length2 the length of the second input vector
 * @param packing true for column-wise, false for row-wise
 * @return encrypted product result
 */
Ciphertext<DCRTPoly> vectorTransposeVectorMultCC(
    const CryptoContext<DCRTPoly> &cryptoContext,
    Ciphertext<DCRTPoly> vector1C,
    Ciphertext<DCRTPoly> vector2C,
    const size_t length1,
    const size_t length2,
    const bool packing
);
