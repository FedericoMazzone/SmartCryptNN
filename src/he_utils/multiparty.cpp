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

/*
    Multiparty HE.
*/

#define PROFILE

#include "multiparty.h"

#include "../he_utils/matrix_multiplication.h"
#include "../vector_utils/vector_utils.h"
#include "misc.h"

#include <cassert>

#define min(a,b) (((a)<(b))?(a):(b))
#define max(a,b) (((a)>(b))?(a):(b))



Party::Party(
    const usint id
)
{
    this->id = id;
}


void Party::importCryptoContext(
    std::stringstream &cryptoContextSerialized
)
{

    cryptoContextSerialized.seekg(0, std::ios::beg);

    // CryptoContext<DCRTPoly> clientCC;
    // clientCC->ClearEvalMultKeys();
    // clientCC->ClearEvalAutomorphismKeys();
    // lbcrypto::CryptoContextFactory<lbcrypto::DCRTPoly>::ReleaseAllContexts();
    // Serial::Deserialize(clientCC, cryptoContextSerialized, SerType::BINARY);
    // this->cryptoContext = clientCC;

    Serial::Deserialize(this->cryptoContext, cryptoContextSerialized, SerType::BINARY);

}


void Party::generatePrivateKeyShare()
{

    const auto cryptoParams = std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(this->cryptoContext->GetCryptoParameters());
    const auto elementParams = cryptoParams->GetElementParams();
    const auto paramsPK = cryptoParams->GetParamsPK();

    const DCRTPoly::DggType& dgg = cryptoParams->GetDiscreteGaussianGenerator();
    DCRTPoly::TugType tug;

    DCRTPoly s;
    switch (cryptoParams->GetSecretKeyDist())
    {
        case GAUSSIAN:
            s = DCRTPoly(dgg, paramsPK, Format::EVALUATION);
            break;
        case UNIFORM_TERNARY:
            s = DCRTPoly(tug, paramsPK, Format::EVALUATION);
            break;
        case SPARSE_TERNARY:
            s = DCRTPoly(tug, paramsPK, Format::EVALUATION, 16);
            break;
        default:
            break;
    }

    this->privateKey = std::make_shared<PrivateKeyImpl<DCRTPoly>>(this->cryptoContext);
    this->privateKey->SetPrivateElement(std::move(s));

}


DCRTPoly Party::generatePublicKeyShare(
    DCRTPoly commonPolynomial
)
{

    const auto cryptoParams = std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(this->cryptoContext->GetCryptoParameters());
    const auto elementParams = cryptoParams->GetElementParams();
    const auto paramsPK = cryptoParams->GetParamsPK();

    const auto ns = cryptoParams->GetNoiseScale();
    const DCRTPoly::DggType& dgg = cryptoParams->GetDiscreteGaussianGenerator();
    DCRTPoly e(dgg, paramsPK, Format::EVALUATION);

    DCRTPoly s = this->privateKey->GetPrivateElement();

    // pks = e - a * s
    DCRTPoly publicKeyShare = ns * e - commonPolynomial * s;

    usint sizeQ  = elementParams->GetParams().size();
    usint sizePK = paramsPK->GetParams().size();
    if (sizePK > sizeQ)
    {
        s.DropLastElements(sizePK - sizeQ);
        this->privateKey->SetPrivateElement(std::move(s));
    }

    return publicKeyShare;

}


EvalKey<DCRTPoly> Party::generateRelinearizationKey1(
    EvalKey<DCRTPoly> commonPolyAsEvalKey
)
{

    return this->cryptoContext->MultiKeySwitchGen
    (
        this->privateKey,
        this->privateKey,
        commonPolyAsEvalKey
    );

}


EvalKey<DCRTPoly> Party::generateRelinearizationKey2(
    EvalKey<DCRTPoly> relinearizationKey1
)
{

    return this->cryptoContext->MultiMultEvalKey
    (
        this->privateKey,
        relinearizationKey1
    );

}


void Party::insertRelinearizationKey(
    EvalKey<DCRTPoly> relinearizationKey
)
{

    this->cryptoContext->InsertEvalMultKey(
        {relinearizationKey}
    );

}


std::shared_ptr<std::map<usint, EvalKey<DCRTPoly>>> Party::generateRotationKeysShare(
    const std::shared_ptr<std::map<usint, EvalKey<DCRTPoly>>> zeroEvalRotateKeys,
    const std::vector<int32_t> &indices
)
{
    
    return this->cryptoContext->MultiEvalAtIndexKeyGen(
        this->privateKey,
        zeroEvalRotateKeys,
        indices
    );

}


void Party::insertRotationKeys(
    std::shared_ptr<std::map<usint, EvalKey<DCRTPoly>>> rotationKeys
)
{

    this->cryptoContext->InsertEvalAutomorphismKey(
        rotationKeys
    );

}


Ciphertext<DCRTPoly> Party::computeDecryptionShare(
    const Ciphertext<DCRTPoly> &ciphertext
)
{

    return this->cryptoContext->MultipartyDecryptMain(
        {ciphertext},
        this->privateKey
    )[0];

}


std::vector<Ciphertext<DCRTPoly>> Party::computeBootstrappingShare(
    const Ciphertext<DCRTPoly> &ciphertext1,
    Ciphertext<DCRTPoly> commonPolynomial
)
{

    // Content omitted until OpenFHE threshold-CKKS bootstrapping is publicly available.

}


CryptoContext<DCRTPoly> generateCryptoContext(
    const usint integralPrecision,
    const usint decimalPrecision,
    const usint multiplicativeDepth,
    const usint batchSize,
    const bool verbose,
    std::string *verboseLog
)
{

    const usint plaintextPrecision = integralPrecision + decimalPrecision;

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetSecretKeyDist(UNIFORM_TERNARY);
    parameters.SetSecurityLevel(HEStd_128_classic);
    parameters.SetScalingModSize(decimalPrecision);
    parameters.SetScalingTechnique(ScalingTechnique::FLEXIBLEAUTO);
    parameters.SetFirstModSize(plaintextPrecision);
    parameters.SetMultiplicativeDepth(multiplicativeDepth);
    parameters.SetKeySwitchTechnique(KeySwitchTechnique::HYBRID);
    parameters.SetBatchSize(batchSize);
    
    CryptoContext<DCRTPoly> cryptoContext = GenCryptoContext(parameters);
    cryptoContext->Enable(PKE);
    cryptoContext->Enable(KEYSWITCH);
    cryptoContext->Enable(LEVELEDSHE);
    cryptoContext->Enable(ADVANCEDSHE);
    cryptoContext->Enable(MULTIPARTY);

    if (verbose || verboseLog)
    {
        const BigInteger ciphertextModulus = cryptoContext->GetModulus();
        const usint ciphertextModulusBitsize = ciphertextModulus.GetLengthForBase(2);
        const usint ringDimension = cryptoContext->GetRingDimension();
        const usint maxNumSlots = ringDimension / 2;
        const auto elementParameters = cryptoContext->GetCryptoParameters()->GetElementParams()->GetParams();
        std::vector<NativeInteger> moduliChain(elementParameters.size());
        std::vector<usint> moduliChainBitsize(elementParameters.size());
        for (size_t i = 0; i < elementParameters.size(); i++)
        {
            moduliChain[i] = elementParameters[i]->GetModulus();
            moduliChainBitsize[i] = moduliChain[i].GetLengthForBase(2);
        }

        std::ostringstream logMessage;
        logMessage << "SETUP PARAMETERS"                                                             << std::endl;
        logMessage << "Integral Bit Precision        : " << integralPrecision                        << std::endl;
        logMessage << "Decimal Bit Precision         : " << decimalPrecision                         << std::endl;
        logMessage << "Ciphertext Modulus Precision  : " << ciphertextModulusBitsize                 << std::endl;
        logMessage << "Ring Dimension                : " << ringDimension                            << std::endl;
        logMessage << "Max Slots                     : " << maxNumSlots                              << std::endl;
        logMessage << "Slots                         : " << batchSize                                << std::endl;
        logMessage << "Multiplicative Depth          : " << parameters.GetMultiplicativeDepth()      << std::endl;
        logMessage << "Security Level                : " << parameters.GetSecurityLevel()            << std::endl;
        logMessage << "Secret Key Distribution       : " << parameters.GetSecretKeyDist()            << std::endl;
        logMessage << "Scaling Technique             : " << parameters.GetScalingTechnique()         << std::endl;
        logMessage << "Encryption Technique          : " << parameters.GetEncryptionTechnique()      << std::endl;
        logMessage << "Multiplication Technique      : " << parameters.GetMultiplicationTechnique()  << std::endl;
        logMessage << "Moduli Chain Bitsize          : " << moduliChainBitsize                       << std::endl;
        logMessage << "Moduli Chain                  : " << moduliChain                              << std::endl;
        logMessage << std::endl;

        if (verbose)
            std::cout << logMessage.str();

        if (verboseLog)
            *verboseLog = logMessage.str();

    }

    return cryptoContext;

}


PublicKey<DCRTPoly> aggregatePublicKeyShares(
    const CryptoContext<DCRTPoly> &cryptoContext,
    const std::vector<DCRTPoly> &publicKeyShares,
    const DCRTPoly commonPolynomial
)
{

    const auto paramsPK = cryptoContext->GetCryptoParameters()->GetParamsPK();

    DCRTPoly b = DCRTPoly(paramsPK, Format::EVALUATION, true);
    for (const DCRTPoly &publicKeyShare : publicKeyShares)
        b += publicKeyShare;
    
    PublicKey<DCRTPoly> publicKey(std::make_shared<PublicKeyImpl<DCRTPoly>>(cryptoContext));
    publicKey->SetPublicElementAtIndex(0, std::move(b));
    publicKey->SetPublicElementAtIndex(1, std::move(commonPolynomial));

    return publicKey;

}


std::vector<double> decrypt(
    const CryptoContext<DCRTPoly> &cryptoContext,
    const std::vector<Party> &parties,
    const Ciphertext<DCRTPoly> &ciphertext,
    const size_t length
)
{
    Plaintext plaintext = protocol::decrypt(
        cryptoContext,
        parties,
        ciphertext
    );

    plaintext->SetLength(length);

    return plaintext->GetRealPackedValue();
}


std::vector<std::vector<double>> decryptMatrix(
    const CryptoContext<DCRTPoly> &cryptoContext,
    const std::vector<Party> &parties,
    const Ciphertext<DCRTPoly> &matrixC,
    const size_t ogNumRows,
    const size_t ogNumCols,
    const bool packing,
    const size_t numRowsPrevMatrix
)
{
    Plaintext matrixP = protocol::decrypt(
        cryptoContext,
        parties,
        matrixC
    );

    std::vector<std::vector<double>> matrix = decodeMatrix(
        matrixP,
        ogNumRows,
        ogNumCols,
        packing,
        numRowsPrevMatrix
    );

    return matrix;
}


std::vector<double> decryptVector(
    const CryptoContext<DCRTPoly> &cryptoContext,
    const std::vector<Party> &parties,
    const Ciphertext<DCRTPoly> &vectorC,
    const size_t ogLength,
    const bool packing,
    const size_t numRowsPrevMatrix
)
{
    if (packing)
    {
        return decrypt(
            cryptoContext,
            parties,
            vectorC,
            ogLength
        );
    }
    else
    {
        return transpose(decryptMatrix(
            cryptoContext,
            parties,
            vectorC,
            ogLength,
            1,
            false,
            numRowsPrevMatrix
        ))[0];
    }
}


void printDecryption(
    const CryptoContext<DCRTPoly> &cryptoContext,
    const std::vector<Party> &parties,
    const Ciphertext<DCRTPoly> &ciphertext,
    const size_t length,
    const std::string label,
    const double threshold,
    const std::vector<double> expectedResult,
    const size_t numCols
)
{

    Plaintext plaintext;
    const size_t ciphertextLevel = ciphertext->GetLevel();

    try
    {
        plaintext = protocol::decrypt(
            cryptoContext,
            parties,
            ciphertext
        );

        plaintext->SetLength(length);
        const double precision = plaintext->GetLogPrecision();

        std::vector<double> result = plaintext->GetRealPackedValue();

        if (threshold > 0.0)
            for (size_t i = 0; i < result.size(); i++)
                if (abs(result[i]) < threshold)
                    result[i] = 0.0;

        if (numCols > 0)
        {
            std::vector<double> column;
            for (size_t i = 0; i < 1.0 * result.size() / numCols; i++)
                column.push_back(result[i * numCols]);
            result = column;
        }

        std::cout << "Decryption " << label << " [level " << ciphertextLevel
                  << "] [precision " << precision << " bits] " << result
                  << std::endl;
        
        if (expectedResult.size() > 0)
        {
            double maxError = 0.0;
            for (size_t i = 0; i < expectedResult.size(); i++)
                maxError = max(maxError, abs(expectedResult[i] - result[i]));
            std::cout << "Error " << maxError << std::endl;
        }
    }
    catch (const std::exception& e)
    {
        std::cout << "Decryption " << label << " [level " << ciphertextLevel
                  << "] failed" << std::endl;
    }

}


// Protocols

CryptoContext<DCRTPoly> protocol::setup(
    const usint integralPrecision,
    const usint decimalPrecision,
    const usint multiplicativeDepth,
    const usint batchSize,
    std::vector<Party> &parties,
    const bool verbose
)
{

    CryptoContext<DCRTPoly> cryptoContext = generateCryptoContext(
        integralPrecision,
        decimalPrecision,
        multiplicativeDepth,
        batchSize,
        verbose
    );

    // for (usint i = 0; i < numParties; i++)
    //     parties[i].cryptoContext = cryptoContext;

    // Central server serializes the cryptoContext.
    std::stringstream cryptoContextSerialized;
    Serial::Serialize(cryptoContext, cryptoContextSerialized, SerType::BINARY);
    if (verbose)
        std::cout << "Crypto context serialized - size: " << 
                    cryptoContextSerialized.str().length() << std::endl;

    // Central server sends the cryptoContext to EdgeNodes.
    for (Party &party : parties)
    {
        ////////////////////////////////////////////////////////
        // Central Server > Parties [ cryptoContext ]
        ////////////////////////////////////////////////////////

        party.importCryptoContext(cryptoContextSerialized);
        if (verbose)
            std::cout << "Party " << party.id
                    << " successfully imported cryptoContext" << std::endl;
    }
    std::cout << std::endl;

    return cryptoContext;

}


PublicKey<DCRTPoly> protocol::keyGeneration(
    const CryptoContext<DCRTPoly> cryptoContext,
    std::vector<Party> &parties,
    const bool verbose
)
{

    if (verbose)
        std::cout << "Private/public key generation protocol...     ";

    DCRTPoly commonPolynomial = generateRandomPolynomialPK(cryptoContext);

    // Serialize common polynomial.
    std::stringstream commonPolynomialSerialized;
    Serial::Serialize(commonPolynomial, commonPolynomialSerialized, SerType::BINARY);

    std::vector<DCRTPoly> publicKeyShares;
    // DCRTPoly publicKeyShare;
    for (Party &party : parties)
    {
        // Generate the private key shares.
        party.generatePrivateKeyShare();

        ////////////////////////////////////////////////////////
        // Central Server > Parties [ commonPolynomial ]
        ////////////////////////////////////////////////////////

        // Deserialize common polynomial.
        DCRTPoly commonPolynomialReceived;
        commonPolynomialSerialized.seekg(0, std::ios::beg);
        Serial::Deserialize(commonPolynomialReceived, commonPolynomialSerialized, SerType::BINARY);

        // Generate the public key shares.
        DCRTPoly publicKeyShare = party.generatePublicKeyShare(commonPolynomialReceived);
        // publicKeyShares.push_back(publicKeyShare);

        // Serialize public key share.
        std::stringstream publicKeyShareSerialized;
        Serial::Serialize(publicKeyShare, publicKeyShareSerialized, SerType::BINARY);

        ////////////////////////////////////////////////////////
        // Central Server < Parties [ publicKeyShare ]
        ////////////////////////////////////////////////////////

        // Deserialize and collect public key shares.
        DCRTPoly publicKeyShareReceived;
        publicKeyShareSerialized.seekg(0, std::ios::beg);
        Serial::Deserialize(publicKeyShareReceived, publicKeyShareSerialized, SerType::BINARY);

        publicKeyShares.push_back(publicKeyShareReceived);
    }

    // Aggregate the public key shares into the master public key.
    PublicKey<DCRTPoly> publicKey = aggregatePublicKeyShares(
        cryptoContext,
        publicKeyShares,
        commonPolynomial
    );
    // for (Party &party : parties)
    //     party.publicKey = publicKey;

    // Serialize public key.
    std::stringstream publicKeySerialized;
    Serial::Serialize(publicKey, publicKeySerialized, SerType::BINARY);

    for (Party &party : parties)
    {
        ////////////////////////////////////////////////////////
        // Central Server > Parties [ publicKey ]
        ////////////////////////////////////////////////////////

        // Deserialize public key.
        PublicKey<DCRTPoly> publicKeyReceived;
        publicKeySerialized.seekg(0, std::ios::beg);
        Serial::Deserialize(publicKeyReceived, publicKeySerialized, SerType::BINARY);

        party.publicKey = publicKeyReceived;
    }

    if (verbose)
        std::cout << "COMPLETED" << std::endl << std::endl;
    
    return publicKey;

}


void protocol::relinearizationKeyGeneration(
    const CryptoContext<DCRTPoly> cryptoContext,
    std::vector<Party> &parties,
    const bool verbose
)
{

    if (verbose)
        std::cout << "Relinearization key generation protocol...    ";

    DCRTPoly commonPolynomial = generateRandomPolynomialQP(cryptoContext);
    EvalKey<DCRTPoly> commonPolynomialAsKey = aToEvalKey(
        cryptoContext,
        commonPolynomial
    );

    EvalKey<DCRTPoly> key1 = a0ToEvalKey(
        cryptoContext,
        commonPolynomial
    );
    EvalKey<DCRTPoly> key1Share;
    for (Party &party : parties)
    {
        ////////////////////////////////////////////////////////
        // Central Server > Parties [ commonPolynomialAsKey ]
        ////////////////////////////////////////////////////////

        key1Share = party.generateRelinearizationKey1(
            commonPolynomialAsKey
        );

        ////////////////////////////////////////////////////////
        // Central Server < Parties [ key1Share ]
        ////////////////////////////////////////////////////////

        key1 = cryptoContext->MultiAddEvalKeys(
            key1,
            key1Share
        );
    }

    EvalKey<DCRTPoly> key2 = generateZeroEvalKey(
        cryptoContext
    );
    EvalKey<DCRTPoly> key2Share;
    for (Party &party : parties)
    {
        ////////////////////////////////////////////////////////
        // Central Server > Parties [ key1 ]
        ////////////////////////////////////////////////////////

        key2Share = party.generateRelinearizationKey2(
            key1
        );

        ////////////////////////////////////////////////////////
        // Central Server < Parties [ key2Share ]
        ////////////////////////////////////////////////////////

        key2 = cryptoContext->MultiAddEvalMultKeys(
            key2,
            key2Share
        );
    }

    for (Party &party : parties)
    {
        ////////////////////////////////////////////////////////
        // Central Server > Parties [ key2 ]
        ////////////////////////////////////////////////////////

        party.insertRelinearizationKey(key2);
    }

    cryptoContext->InsertEvalMultKey({key2});

    if (verbose)
        std::cout << "COMPLETED" << std::endl << std::endl;

}


void protocol::rotationKeyGeneration(
    const CryptoContext<DCRTPoly> cryptoContext,
    std::vector<Party> &parties,
    const std::vector<int32_t> &indices,
    const bool verbose
)
{

    if (verbose)
        std::cout << "Rotation keys generation protocol...          ";

    std::shared_ptr<std::map<usint, EvalKey<DCRTPoly>>> zeroEvalRotateKeys =
        generateZeroRotateKeys(
            cryptoContext,
            indices
        );

    std::shared_ptr<std::map<usint, EvalKey<DCRTPoly>>> evalRotateKeys =
        zeroEvalRotateKeys;
    std::shared_ptr<std::map<usint, EvalKey<DCRTPoly>>> evalRotateKeysShare;
    for (Party party : parties)
    {
        ////////////////////////////////////////////////////////
        // Central Server > Parties [ zeroEvalRotateKeys, indices ]
        ////////////////////////////////////////////////////////

        evalRotateKeysShare = party.generateRotationKeysShare(
            zeroEvalRotateKeys,
            indices
        );

        ////////////////////////////////////////////////////////
        // Central Server < Parties [ evalRotateKeysShare ]
        ////////////////////////////////////////////////////////

        evalRotateKeys = cryptoContext->MultiAddEvalAutomorphismKeys(
            evalRotateKeys,
            evalRotateKeysShare
        );
    }

    for (Party party : parties)
    {
        ////////////////////////////////////////////////////////
        // Central Server > Parties [ evalRotateKeys ]
        ////////////////////////////////////////////////////////

        party.insertRotationKeys(evalRotateKeys);
    }
    
    cryptoContext->InsertEvalAutomorphismKey(evalRotateKeys);

    if (verbose)
        std::cout << "COMPLETED" << std::endl << std::endl;

}


Plaintext protocol::decrypt(
    const CryptoContext<DCRTPoly> &cryptoContext,
    const std::vector<Party> &parties,
    const Ciphertext<DCRTPoly> &ciphertext
)
{

    Plaintext result;
    std::vector<Ciphertext<DCRTPoly>> decryptionShares;

    ////////////////////////////////////////////////////////
    // Central Server < Querier Party [ ciphertext ]
    ////////////////////////////////////////////////////////

    // c0
    decryptionShares.push_back(ciphertext);

    Ciphertext<DCRTPoly> decryptionShare;
    for (Party party : parties)
    {
        ////////////////////////////////////////////////////////
        // Central Server > Other Parties [ ciphertext ]
        ////////////////////////////////////////////////////////

        // s * c1 + e
        decryptionShare = party.computeDecryptionShare(ciphertext);
        decryptionShares.push_back(decryptionShare);

        ////////////////////////////////////////////////////////
        // Central Server < Other Parties [ decryptionShare ]
        ////////////////////////////////////////////////////////
    }

    ////////////////////////////////////////////////////////
    // Central Server > Querier Party [ decryptionShares ]
    // note that the central server can sum the all but one decryption shares
    ////////////////////////////////////////////////////////

    cryptoContext->MultipartyDecryptFusion(decryptionShares, &result);

    return result;

}


Ciphertext<DCRTPoly> protocol::bootstrap(
    const CryptoContext<DCRTPoly> &cryptoContext,
    const PublicKey<DCRTPoly> &publicKey,
    const std::vector<Party> &parties,
    Ciphertext<DCRTPoly> ciphertext
)
{

    // Content omitted until OpenFHE threshold-CKKS bootstrapping is publicly available.

}
