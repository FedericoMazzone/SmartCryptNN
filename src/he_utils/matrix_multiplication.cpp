/*
    Matrix multiplication.
*/

#include "matrix_multiplication.h"

#include "../vector_utils/vector_utils.h"

#include <iostream>


/**
 * Compute the inner product between two encrypted vectors.
 * @param cryptoContext the crypto context
 * @param publicKey the public key
 * @param vector1C first encrypted input vector
 * @param vector2C second encrypted input vector
 * @param vectorLength length of the vector (in plaintext)
 * @param masking whether you want the output ciphertext to contain only the
 * output value in the first position, and 0s in the other positions
 * @return encrypted inner product value
 */
Ciphertext<DCRTPoly> innerProductCC(
    CryptoContext<DCRTPoly> cryptoContext,
    PublicKey<DCRTPoly> publicKey,
    Ciphertext<DCRTPoly> vector1C,
    Ciphertext<DCRTPoly> vector2C,
    size_t vectorLength,
    bool masking = false
)
{
    Ciphertext<DCRTPoly> v1v2C = cryptoContext->EvalMult(vector1C, vector2C);

    const std::vector<int64_t> ZERO = {0};
    const Plaintext ZERO_PLAINTEXT = cryptoContext->MakePackedPlaintext(ZERO);
    Ciphertext<DCRTPoly> innerProductC = cryptoContext->Encrypt(publicKey, ZERO_PLAINTEXT);
    for (size_t i = 0; i < vectorLength; i++)
        innerProductC = cryptoContext->EvalAdd(innerProductC, cryptoContext->EvalRotate(v1v2C, i));
    
    if (masking) {
        const std::vector<int64_t> ONE = {1};
        const Plaintext ONE_PLAINTEXT = cryptoContext->MakePackedPlaintext(ONE);
        innerProductC = cryptoContext->EvalMult(innerProductC, ONE_PLAINTEXT);
    }

    return innerProductC;
}


/**
 * Compute the inner product between an encrypted vector and a plaintext vector.
 * The naive algorithm is used.
 * @param cryptoContext the crypto context
 * @param publicKey the public key
 * @param vector1C first encrypted input vector
 * @param vector2 second (plaintext) input vector
 * @param masking whether you want the output ciphertext to contain only the
 * output value in the first position, and 0s in the other positions
 * @return encrypted inner product value
 */
Ciphertext<DCRTPoly> innerProductCP(
    CryptoContext<DCRTPoly> cryptoContext,
    PublicKey<DCRTPoly> publicKey,
    Ciphertext<DCRTPoly> vector1C,
    std::vector<int64_t> vector2,
    bool masking = false
)
{
    Plaintext vector2P  = cryptoContext->MakePackedPlaintext(vector2);

    Ciphertext<DCRTPoly> v1v2C = cryptoContext->EvalMult(vector1C, vector2P);

    const std::vector<int64_t> ZERO = {0};
    const Plaintext ZERO_PLAINTEXT = cryptoContext->MakePackedPlaintext(ZERO);
    Ciphertext<DCRTPoly> innerProductC = cryptoContext->Encrypt(publicKey, ZERO_PLAINTEXT);
    for (size_t i = 0; i < vector2.size(); i++)
        innerProductC = cryptoContext->EvalAdd(innerProductC, cryptoContext->EvalRotate(v1v2C, i));

    if (masking) {
        const std::vector<int64_t> ONE = {1};
        const Plaintext ONE_PLAINTEXT = cryptoContext->MakePackedPlaintext(ONE);
        innerProductC = cryptoContext->EvalMult(innerProductC, ONE_PLAINTEXT);
    }

    return innerProductC;
}


/**
 * Compute the inner product between an encrypted vector and a plaintext vector.
 * The recursive vector sum-up is used.
 * @param cryptoContext the crypto context
 * @param vector1C first encrypted input vector
 * @param vector2 second (plaintext) input vector
 * @param masking whether you want the output ciphertext to contain only the
 * output value in the first position, and 0s in the other positions
 * @return encrypted inner product value
 */
Ciphertext<DCRTPoly> innerProductFastCP(
    CryptoContext<DCRTPoly> cryptoContext,
    Ciphertext<DCRTPoly> vector1C,
    std::vector<int64_t> vector2,
    bool masking = false
)
{
    Plaintext vector2P  = cryptoContext->MakePackedPlaintext(vector2);

    Ciphertext<DCRTPoly> v1v2C = cryptoContext->EvalMult(vector1C, vector2P);

    for (size_t i = 0; i < log2(vector2.size()); i++)
        v1v2C = cryptoContext->EvalAdd(v1v2C, cryptoContext->EvalRotate(v1v2C, 1 << i));

    if (masking) {
        const std::vector<int64_t> ONE = {1};
        const Plaintext ONE_PLAINTEXT = cryptoContext->MakePackedPlaintext(ONE);
        v1v2C = cryptoContext->EvalMult(v1v2C, ONE_PLAINTEXT);
    }

    return v1v2C;
}


/**
 * Compute the product between an encrypted vector and a plaintext matrix.
 * The naive algorithm with the naive inner product implementation is used.
 * The output is automatically masked.
 * @param cryptoContext the crypto context
 * @param publicKey the public key
 * @param vectorC encrypted input vector
 * @param matrix (plaintext) input matrix
 * @return encrypted vector-matrix product
 */
Ciphertext<DCRTPoly> vectorMatrixMultByInnProdCP(
    CryptoContext<DCRTPoly> cryptoContext,
    PublicKey<DCRTPoly> publicKey,
    Ciphertext<DCRTPoly> vectorC,
    std::vector<std::vector<int64_t>> matrix
)
{
    const std::vector<int64_t> ZERO = {0};
    const Plaintext ZERO_PLAINTEXT = cryptoContext->MakePackedPlaintext(ZERO);
    Ciphertext<DCRTPoly> result = cryptoContext->Encrypt(publicKey, ZERO_PLAINTEXT);
    std::vector<std::vector<int64_t>> matrixT = transpose(matrix);
    Ciphertext<DCRTPoly> innProdC;
    for (size_t i = 0; i < matrixT.size(); i++) {
        innProdC = innerProductCP(cryptoContext, publicKey, vectorC, matrixT[i], true);
        innProdC = cryptoContext->EvalRotate(innProdC, -i);
        result = cryptoContext->EvalAdd(result, innProdC);
    }
    return result;
}


/**
 * Compute the product between an encrypted vector and a plaintext matrix.
 * The naive algorithm with the recursive-sum inner product implementation is
 * used.
 * The output is automatically masked.
 * @param cryptoContext the crypto context
 * @param publicKey the public key
 * @param vectorC encrypted input vector
 * @param matrix (plaintext) input matrix
 * @return encrypted vector-matrix product
 */
Ciphertext<DCRTPoly> vectorMatrixMultByInnProdFastCP(
    CryptoContext<DCRTPoly> cryptoContext,
    PublicKey<DCRTPoly> publicKey,
    Ciphertext<DCRTPoly> vectorC,
    std::vector<std::vector<int64_t>> matrix
)
{
    const std::vector<int64_t> ZERO = {0};
    const Plaintext ZERO_PLAINTEXT = cryptoContext->MakePackedPlaintext(ZERO);
    Ciphertext<DCRTPoly> result = cryptoContext->Encrypt(publicKey, ZERO_PLAINTEXT);
    std::vector<std::vector<int64_t>> matrixT = transpose(matrix);
    Ciphertext<DCRTPoly> innProdC;
    for (size_t i = 0; i < matrixT.size(); i++) {
        innProdC = innerProductFastCP(cryptoContext, vectorC, matrixT[i], true);
        innProdC = cryptoContext->EvalRotate(innProdC, -i);
        result = cryptoContext->EvalAdd(result, innProdC);
    }
    return result;
}


Ciphertext<DCRTPoly> vectorMatrixMultPackCP(
    CryptoContext<DCRTPoly> cryptoContext,
    PublicKey<DCRTPoly> publicKey,
    Ciphertext<DCRTPoly> vectorC,
    std::vector<std::vector<int64_t>> matrix,
    bool packing,
    int numRowsPrevMatrix,
    bool masking,
    bool transposing
)
{
    // Store original matrix size.
    size_t ogNumRows = matrix.size();
    size_t ogNumCols = matrix[0].size();

    // Pad and flatten the matrix.
    size_t numRows = nextPowerOf2(ogNumRows);
    size_t numCols = packing ? nextPowerOf2(ogNumCols) : nextPowerOf2(numRowsPrevMatrix);
    matrix = resizeMatrix(matrix, numRows, numCols);
    std::vector<int64_t> matrixFlat = flattenMatrix(matrix, !packing);
    Plaintext matrixFlatP = cryptoContext->MakePackedPlaintext(matrixFlat);

    // Pad and repeat the vector.
    for (size_t i = 0; i < log2(ogNumCols); i++)
        vectorC = cryptoContext->EvalAdd(vectorC, cryptoContext->EvalRotate(vectorC, -((packing ? numRows : 1) << i)));
    
    // Multiply and sum (the result is stored in the first row of the matrix).
    Ciphertext<DCRTPoly> prod = cryptoContext->EvalMult(vectorC, matrixFlatP);
    for (size_t i = 0; i < log2(numRows); i++)
        prod = cryptoContext->EvalAdd(prod, cryptoContext->EvalRotate(prod, (packing ? 1 : numCols) << i));

    // Mask out the result.
    if (!(packing && transposing) && masking) {
        std::vector<int64_t> mask;
        if (packing) {
            for (size_t i = 0; i < numCols; i++)
                for (size_t j = 0; j < numRows; j++)
                    if (j == 0 && i < ogNumCols)
                        mask.push_back(1);
                    else
                        mask.push_back(0);
        } else {
            mask.insert(mask.end(), ogNumCols, 1);
        }
        Plaintext maskP = cryptoContext->MakePackedPlaintext(mask);
        prod = cryptoContext->EvalMult(prod, maskP);
    }

    // Transpose the result.
    // TODO: improve transposition (easy if rows >= cols)
    if (packing && transposing) {
        const std::vector<int64_t> ZERO = {0};
        const Plaintext ZERO_PLAINTEXT = cryptoContext->MakePackedPlaintext(ZERO);
        Ciphertext<DCRTPoly> res = cryptoContext->Encrypt(publicKey, ZERO_PLAINTEXT);
        std::vector<int64_t> mask = {1};
        Plaintext maskP;
        for (size_t i = 0; i < ogNumCols; i++) {
            maskP = cryptoContext->MakePackedPlaintext(mask);
            res = cryptoContext->EvalAdd(
                    res,
                    cryptoContext->EvalMult(
                        cryptoContext->EvalRotate(
                            prod,
                            i * (numRows - 1)),
                        maskP));
            mask.insert(mask.begin(), 0);
        }
        prod = res;
    }

    return prod;
}


std::vector<double> encodeMatrix(
    std::vector<std::vector<double>> matrix,
    size_t &numRows,
    size_t &numCols,
    const bool packing,
    const size_t numRowsPrevMatrix
)
{
    size_t ogNumRows = matrix.size();
    size_t ogNumCols = matrix[0].size();

    numRows = nextPowerOf2(ogNumRows);
    numCols = packing ? nextPowerOf2(ogNumCols) : nextPowerOf2(numRowsPrevMatrix);

    matrix = resizeMatrix(matrix, numRows, numCols);

    std::vector<double> matrixFlat = flattenMatrix(matrix, !packing);

    return matrixFlat;
}


Plaintext encodeMatrix(
    CryptoContext<DCRTPoly> cryptoContext,
    std::vector<std::vector<double>> matrix,
    size_t &numRows,
    size_t &numCols,
    const bool packing,
    const size_t numRowsPrevMatrix
)
{
    std::vector<double> matrixFlat = encodeMatrix(
        matrix,
        numRows,
        numCols,
        packing,
        numRowsPrevMatrix
    );

    Plaintext matrixFlatP = cryptoContext->MakeCKKSPackedPlaintext(matrixFlat);

    return matrixFlatP;
}


std::vector<std::vector<double>> decodeMatrix(
    const Plaintext &matrixP,
    const size_t ogNumRows,
    const size_t ogNumCols,
    const bool packing,
    const size_t numRowsPrevMatrix
)
{
    size_t numRows = nextPowerOf2(ogNumRows);
    size_t numCols = packing ? nextPowerOf2(ogNumCols) : nextPowerOf2(numRowsPrevMatrix);
    
    matrixP->SetLength(numRows * numCols);
    std::vector<double> matrixFlat = matrixP->GetRealPackedValue();

    std::vector<std::vector<double>> matrix(ogNumRows, std::vector<double>(ogNumCols));
    for (size_t i = 0; i < ogNumRows; i++)
        for (size_t j = 0; j < ogNumCols; j++)
            matrix[i][j] = matrixFlat[packing ? (j * numRows + i) : (i * numCols + j)];

    return matrix;
}


std::vector<double> decodeVector(
    const Plaintext &vectorP,
    const size_t ogLength,
    const bool packing,
    const size_t numRowsPrevMatrix
)
{
    if (packing)
    {
        vectorP->SetLength(ogLength);
        return vectorP->GetRealPackedValue();
    }
    else
    {
        return transpose(decodeMatrix(
            vectorP,
            ogLength,
            1,
            false,
            numRowsPrevMatrix
        ))[0];
    }
}


Ciphertext<DCRTPoly> maskVector(
    CryptoContext<DCRTPoly> cryptoContext,
    Ciphertext<DCRTPoly> ciphertext,
    bool packing,
    size_t ogNumCols,
    size_t numRows,
    size_t numCols
)
{
    std::vector<double> mask;
    if (packing) 
        mask.insert(mask.end(), ogNumCols, 1.0);
    else
    {
        for (size_t i = 0; i < numCols; i++)
            for (size_t j = 0; j < numRows; j++)
                if (j == 0 && i < ogNumCols)
                    mask.push_back(1.0);
                else
                    mask.push_back(0.0);
    }
    Plaintext maskP = cryptoContext->MakeCKKSPackedPlaintext(mask);
    return cryptoContext->EvalMult(ciphertext, maskP);
}


Ciphertext<DCRTPoly> vectorMatrixMultPackCP(
    CryptoContext<DCRTPoly> cryptoContext,
    PublicKey<DCRTPoly> publicKey,
    Ciphertext<DCRTPoly> vectorC,
    std::vector<std::vector<double>> matrix,
    bool packing,
    int numRowsPrevMatrix,
    bool masking,
    bool transposing
)
{
    // Store original matrix size.
    size_t ogNumCols = matrix[0].size();

    // Pad and flatten the matrix.
    size_t numRows;
    size_t numCols;
    Plaintext matrixFlatP = encodeMatrix(
        cryptoContext,
        matrix,
        numRows,
        numCols,
        packing,
        numRowsPrevMatrix
    );

    // Pad and repeat the vector.
    for (size_t i = 0; i < log2(ogNumCols); i++)
        vectorC = cryptoContext->EvalAdd(vectorC, cryptoContext->EvalRotate(vectorC, -((packing ? numRows : 1) << i)));
    
    // Multiply and sum (the result is stored in the first row of the matrix).
    Ciphertext<DCRTPoly> prod = cryptoContext->EvalMult(vectorC, matrixFlatP);
    for (size_t i = 0; i < log2(numRows); i++)
        prod = cryptoContext->EvalAdd(prod, cryptoContext->EvalRotate(prod, (packing ? 1 : numCols) << i));

    // Mask out the result.
    if (!(packing && transposing) && masking)
        prod = maskVector(cryptoContext, prod, !packing, ogNumCols, numRows, numCols);

    // Transpose the result.
    // TODO: improve transposition (easy if rows >= cols)
    if (packing && transposing) {
        const std::vector<double> ZERO = {0.0};
        const Plaintext ZERO_PLAINTEXT = cryptoContext->MakeCKKSPackedPlaintext(ZERO);
        Ciphertext<DCRTPoly> res = cryptoContext->Encrypt(publicKey, ZERO_PLAINTEXT);
        std::vector<double> mask = {1.0};
        Plaintext maskP;
        for (size_t i = 0; i < ogNumCols; i++) {
            maskP = cryptoContext->MakeCKKSPackedPlaintext(mask);
            res = cryptoContext->EvalAdd(
                    res,
                    cryptoContext->EvalMult(
                        cryptoContext->EvalRotate(
                            prod,
                            i * (numRows - 1)),
                        maskP));
            mask.insert(mask.begin(), 0);
        }
        prod = res;
    }

    return prod;
}


Ciphertext<DCRTPoly> vectorMatrixMultPackCC(
    CryptoContext<DCRTPoly> cryptoContext,
    PublicKey<DCRTPoly> publicKey,
    Ciphertext<DCRTPoly> vectorC,
    Ciphertext<DCRTPoly> matrixC,
    size_t ogNumCols,
    size_t numRows,
    size_t numCols,
    bool packing,
    bool masking,
    bool transposing
)
{
    // Pad and repeat the vector.
    for (size_t i = 0; i < log2(ogNumCols); i++)
        vectorC = cryptoContext->EvalAdd(vectorC, cryptoContext->EvalRotate(vectorC, -((packing ? numRows : 1) << i)));
    
    // Multiply and sum (the result is stored in the first row of the matrix).
    Ciphertext<DCRTPoly> prod = cryptoContext->EvalMult(vectorC, matrixC);
    for (size_t i = 0; i < log2(numRows); i++)
        prod = cryptoContext->EvalAdd(prod, cryptoContext->EvalRotate(prod, (packing ? 1 : numCols) << i));

    // Mask out the result.
    if (!(packing && transposing) && masking)
        prod = maskVector(cryptoContext, prod, !packing, ogNumCols, numRows, numCols);

    // Transpose the result.
    // TODO: improve transposition (easy if rows >= cols)
    if (packing && transposing) {
        const std::vector<double> ZERO = {0.0};
        const Plaintext ZERO_PLAINTEXT = cryptoContext->MakeCKKSPackedPlaintext(ZERO);
        Ciphertext<DCRTPoly> res = cryptoContext->Encrypt(publicKey, ZERO_PLAINTEXT);
        std::vector<double> mask = {1.0};
        Plaintext maskP;
        for (size_t i = 0; i < ogNumCols; i++) {
            maskP = cryptoContext->MakeCKKSPackedPlaintext(mask);
            res = cryptoContext->EvalAdd(
                    res,
                    cryptoContext->EvalMult(
                        cryptoContext->EvalRotate(
                            prod,
                            i * (numRows - 1)),
                        maskP));
            mask.insert(mask.begin(), 0);
        }
        prod = res;
    }

    return prod;
}


Ciphertext<DCRTPoly> vectorTransposeVectorMultCC(
    const CryptoContext<DCRTPoly> &cryptoContext,
    Ciphertext<DCRTPoly> vector1C,
    Ciphertext<DCRTPoly> vector2C,
    const size_t length1,
    const size_t length2,
    const bool packing
)
{
    // Pad and repeat the vectors.
    for (size_t i = 0; i < log2(length2); i++)
        vector1C = cryptoContext->EvalAdd(vector1C, cryptoContext->EvalRotate(vector1C, -((packing ? nextPowerOf2(length1) : 1) << i)));
    for (size_t i = 0; i < log2(length1); i++)
        vector2C = cryptoContext->EvalAdd(vector2C, cryptoContext->EvalRotate(vector2C, -((packing ? 1 : nextPowerOf2(length2)) << i)));
    
    // Multiply.
    Ciphertext<DCRTPoly> prod = cryptoContext->EvalMult(vector1C, vector2C);

    return prod;
}


// int main(int argc, char* argv[]) {

//     TimeVar t;
//     double processingTime(0.0);
 
//     CCParams<CryptoContextBFVRNS> parameters;
//     parameters.SetPlaintextModulus(536903681);
//     parameters.SetMultiplicativeDepth(4);
//     parameters.SetMaxRelinSkDeg(3);

//     CryptoContext<DCRTPoly> cryptoContext = GenCryptoContext(parameters);
//     cryptoContext->Enable(PKE);
//     cryptoContext->Enable(KEYSWITCH);
//     cryptoContext->Enable(LEVELEDSHE);
//     cryptoContext->Enable(ADVANCEDSHE);

//     PlaintextModulus p = cryptoContext->GetCryptoParameters()->GetPlaintextModulus();
//     int n = cryptoContext->GetCryptoParameters()->GetElementParams()->GetCyclotomicOrder() / 2;
//     double q = cryptoContext->GetCryptoParameters()->GetElementParams()->GetModulus().ConvertToDouble();
//     std::cout << "Plaintext modulus (p) = " << p << std::endl;
//     std::cout << "Polynomial degree (n) = " << n << std::endl;
//     std::cout << "Ciphertext modulus bitsize (log2 q) = " << log2(q) << std::endl;

//     KeyPair<DCRTPoly> keyPair = cryptoContext->KeyGen();
//     if (!keyPair.good()) {
//         std::cout << "Key generation failed!" << std::endl;
//         exit(1);
//     }

//     cryptoContext->EvalMultKeysGen(keyPair.secretKey);

//     std::cout << "Generating rotation keys... ";
//     std::vector<int32_t> indexList = {};
//     for (int i = -100; i <= 100; i++) indexList.push_back(i);
//     for (int i = 0; i <= 10; i++) {
//         indexList.push_back(1 << i);
//         indexList.push_back(-(1 << i));
//     }
//     cryptoContext->EvalRotateKeyGen(keyPair.secretKey, indexList);
//     std::cout << "DONE" << std::endl;

//     std::cout << std::endl;

//     ////////////////////////////////////////////////////////////
//     // Inner product
//     ////////////////////////////////////////////////////////////

//     // const size_t VECTOR_LENGTH = 200;
//     // const int64_t MAX_VALUE = 100;
    
//     // std::vector<int64_t> v1 = genRandVect(VECTOR_LENGTH, MAX_VALUE);
//     // Plaintext v1P  = cryptoContext->MakePackedPlaintext(v1);

//     // std::vector<int64_t> v2 = genRandVect(VECTOR_LENGTH, MAX_VALUE);
//     // Plaintext v2P  = cryptoContext->MakePackedPlaintext(v2);

//     // std::cout << "v1 = " << v1 << std::endl;
//     // std::cout << "v2 = " << v2 << std::endl;

//     // Ciphertext<DCRTPoly> v1C = cryptoContext->Encrypt(keyPair.publicKey, v1P);
//     // Ciphertext<DCRTPoly> v2C = cryptoContext->Encrypt(keyPair.publicKey, v2P);

//     // Ciphertext<DCRTPoly> resC;
//     // Plaintext res;
//     // int64_t resInt64;

//     // TIC(t);
//     // resInt64 = innerProduct(v1, v2);
//     // processingTime = TOC(t);
//     // std::cout << "v1  * v2        = " << resInt64 << " (" << processingTime
//     //           << " ms)" << std::endl;
    
//     // TIC(t);
//     // resC = innerProductCC(cryptoContext, keyPair.publicKey, v1C, v2C, v1.size());
//     // processingTime = TOC(t);
//     // cryptoContext->Decrypt(keyPair.secretKey, resC, &res);
//     // res->SetLength(1);
//     // resInt64 = res->GetPackedValue()[0];
//     // std::cout << "v1C * v2C       = " << resInt64 << " (" << processingTime
//     //           << " ms)" << std::endl;

//     // TIC(t);
//     // resC = innerProductCP(cryptoContext, keyPair.publicKey, v1C, v2);
//     // processingTime = TOC(t);
//     // cryptoContext->Decrypt(keyPair.secretKey, resC, &res);
//     // res->SetLength(1);
//     // resInt64 = res->GetPackedValue()[0];
//     // std::cout << "v1C * v2        = " << resInt64 << " (" << processingTime
//     //           << " ms)" << std::endl;

//     // TIC(t);
//     // resC = innerProductFastCP(cryptoContext, v1C, v2);
//     // processingTime = TOC(t);
//     // cryptoContext->Decrypt(keyPair.secretKey, resC, &res);
//     // res->SetLength(1);
//     // resInt64 = res->GetPackedValue()[0];
//     // std::cout << "v1C * v2 (fast) = " << resInt64 << " (" << processingTime
//     //           << " ms)" << std::endl;

//     ////////////////////////////////////////////////////////////
//     // Vector * matrix
//     ////////////////////////////////////////////////////////////

//     // const size_t ROWS = 5;
//     // const size_t COLS = 3;
//     // const int64_t MAX_VALUE = 100;
    
//     // std::vector<int64_t> vector = genRandVect(ROWS, MAX_VALUE);
//     // Plaintext vectorP  = cryptoContext->MakePackedPlaintext(vector);

//     // std::vector<std::vector<int64_t>> matrix = genRandMatrix(ROWS, COLS, MAX_VALUE);
    
//     // std::cout << "vector = " << vector << std::endl;
//     // std::cout << "matrix = " << matrix << std::endl;

//     // Ciphertext<DCRTPoly> vectorC = cryptoContext->Encrypt(keyPair.publicKey, vectorP);

//     // Ciphertext<DCRTPoly> resC;
//     // Plaintext res;
//     // std::vector<int64_t> resInt64, resInt64tmp;

//     // TIC(t);
//     // resInt64 = vectorMatrixMult(vector, matrix);
//     // processingTime = TOC(t);
//     // std::cout << "vector  * matrix                         = " << resInt64
//     //           << " (" << processingTime << " ms)" << std::endl;

//     // TIC(t);
//     // resC = vectorMatrixMultByInnProdCP(cryptoContext, keyPair.publicKey, vectorC, matrix);
//     // processingTime = TOC(t);
//     // cryptoContext->Decrypt(keyPair.secretKey, resC, &res);
//     // res->SetLength(COLS);
//     // resInt64 = res->GetPackedValue();
//     // std::cout << "vectorC * matrix (by inner product)      = " << resInt64
//     //           << " (" << processingTime << " ms)" << std::endl;
    
//     // TIC(t);
//     // resC = vectorMatrixMultByInnProdFastCP(cryptoContext, keyPair.publicKey, vectorC, matrix);
//     // processingTime = TOC(t);
//     // cryptoContext->Decrypt(keyPair.secretKey, resC, &res);
//     // res->SetLength(COLS);
//     // resInt64 = res->GetPackedValue();
//     // std::cout << "vectorC * matrix (by inner product fast) = " << resInt64
//     //           << " (" << processingTime << " ms)" << std::endl;
    
//     // TIC(t);
//     // resC = vectorMatrixMultPackCP(cryptoContext, keyPair.publicKey, vectorC, matrix);
//     // processingTime = TOC(t);
//     // cryptoContext->Decrypt(keyPair.secretKey, resC, &res);
//     // res->SetLength(COLS);
//     // resInt64 = res->GetPackedValue();
//     // std::cout << "vectorC * matrix (by column packing)     = " << resInt64
//     //           << " (" << processingTime << " ms)" << std::endl;
    
//     // TIC(t);
//     // resC = vectorMatrixMultPackCP(cryptoContext, keyPair.publicKey, vectorC, matrix, true, -1, false, false);
//     // processingTime = TOC(t);
//     // cryptoContext->Decrypt(keyPair.secretKey, resC, &res);
//     // resInt64tmp = res->GetPackedValue();
//     // resInt64.clear();
//     // for (size_t i = 0; i < COLS; i++)
//     //     resInt64.push_back(resInt64tmp[nextPowerOf2(ROWS) * i]);
//     // std::cout << "vectorC * matrix (by column packing noT) = " << resInt64
//     //           << " (" << processingTime << " ms)" << std::endl;
    
//     ////////////////////////////////////////////////////////////
//     // Vector * matrix1 * matrix2
//     ////////////////////////////////////////////////////////////

//     // If you increase the matrix sizes, then remember to also generate more
//     // rotations keys accordingly.
//     const size_t n1 = 5;
//     const size_t n2 = 3;
//     const size_t n3 = 4;
//     const int64_t MAX_VALUE = 100;
    
//     std::vector<int64_t> vector = genRandVect(n1, MAX_VALUE);
//     Plaintext vectorP  = cryptoContext->MakePackedPlaintext(vector);

//     std::vector<std::vector<int64_t>> matrix1 = genRandMatrix(n1, n2, MAX_VALUE);
//     std::vector<std::vector<int64_t>> matrix2 = genRandMatrix(n2, n3, MAX_VALUE);
    
//     std::cout << "vector  = " << vector << std::endl;
//     std::cout << "matrix1 = " << matrix1 << std::endl;
//     std::cout << "matrix2 = " << matrix2 << std::endl;

//     Ciphertext<DCRTPoly> vectorC = cryptoContext->Encrypt(keyPair.publicKey, vectorP);

//     Ciphertext<DCRTPoly> resC;
//     Plaintext res;
//     std::vector<int64_t> resInt64, resInt64tmp;

//     TIC(t);
//     resInt64 = vectorMatrixMult(vector, matrix1);
//     resInt64 = vectorMatrixMult(resInt64, matrix2);
//     processingTime = TOC(t);
//     std::cout << "vector  * matrix1 * matrix2                         = "
//               << resInt64 << " (" << processingTime << " ms)" << std::endl;
    
//     TIC(t);
//     resC = vectorMatrixMultByInnProdCP(cryptoContext, keyPair.publicKey, vectorC, matrix1);
//     resC = vectorMatrixMultByInnProdCP(cryptoContext, keyPair.publicKey, resC, matrix2);
//     processingTime = TOC(t);
//     cryptoContext->Decrypt(keyPair.secretKey, resC, &res);
//     res->SetLength(n3);
//     resInt64 = res->GetPackedValue();
//     std::cout << "vectorC * matrix1 * matrix2 (by inner product)      = "
//               << resInt64 << " (" << processingTime << " ms)" << std::endl;
    
//     TIC(t);
//     resC = vectorMatrixMultByInnProdFastCP(cryptoContext, keyPair.publicKey, vectorC, matrix1);
//     resC = vectorMatrixMultByInnProdFastCP(cryptoContext, keyPair.publicKey, resC, matrix2);
//     processingTime = TOC(t);
//     cryptoContext->Decrypt(keyPair.secretKey, resC, &res);
//     res->SetLength(n3);
//     resInt64 = res->GetPackedValue();
//     std::cout << "vectorC * matrix1 * matrix2 (by inner product fast) = "
//               << resInt64 << " (" << processingTime << " ms)" << std::endl;

//     TIC(t);
//     resC = vectorMatrixMultPackCP(cryptoContext, keyPair.publicKey, vectorC, matrix1);
//     resC = vectorMatrixMultPackCP(cryptoContext, keyPair.publicKey, resC, matrix2);
//     processingTime = TOC(t);
//     cryptoContext->Decrypt(keyPair.secretKey, resC, &res);
//     res->SetLength(n3);
//     resInt64 = res->GetPackedValue();
//     std::cout << "vectorC * matrix1 * matrix2 (by column packing)     = "
//               << resInt64 << " (" << processingTime << " ms)" << std::endl;
    
//     TIC(t);
//     resC = vectorMatrixMultPackCP(cryptoContext, keyPair.publicKey, vectorC, matrix1, true, -1, true, false);
//     resC = vectorMatrixMultPackCP(cryptoContext, keyPair.publicKey, resC, matrix2, false, n1, true, false);
//     processingTime = TOC(t);
//     cryptoContext->Decrypt(keyPair.secretKey, resC, &res);
//     res->SetLength(n3);
//     resInt64 = res->GetPackedValue();
//     std::cout << "vectorC * matrix1 * matrix2 (by alternate packing)  = "
//               << resInt64 << " (" << processingTime << " ms)" << std::endl;

//     return 0;
// }
