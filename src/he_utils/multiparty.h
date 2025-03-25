#pragma once

#include "openfhe.h"

// header files needed for serialization
#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"
#include "key/key-ser.h"
#include "scheme/ckksrns/ckksrns-ser.h"


using namespace lbcrypto;


/**
 * A utility class defining a party that is involved in the collective
 * encryption operation.
 */
class Party
{

private:

    // The secret share of the private key.
    PrivateKey<DCRTPoly> privateKey;

public:

    // Unique party identifier, starting from zero.
    usint id;

    // The common crypto context.
    CryptoContext<DCRTPoly> cryptoContext;

    // The common public key.
    PublicKey<DCRTPoly> publicKey;

    /**
     * Constructor.
     * @param id unique identifier, starting from zero
     */
    Party(
        const usint id
    );


    /**
     * Import a crypto context from a binary string.
     * @param cryptoContextSerialized serialized crypto context
     */
    void importCryptoContext(
        std::stringstream &cryptoContextSerialized
    );


    /**
     * Generate secret share of the private key.
     */
    void generatePrivateKeyShare();


    /**
     * Generate share of the public key.
     * @param commonPolynomial a common polynomial
     * @return public key share
     */
    DCRTPoly generatePublicKeyShare(
        DCRTPoly commonPolynomial
    );


    /**
     * Generate share of the intermediate relinearization key.
     * @param commonPolyAsEvalKey a common polynomial as EvalKey
     * @return intermediate relinearization key share
     */
    EvalKey<DCRTPoly> generateRelinearizationKey1(
        EvalKey<DCRTPoly> commonPolyAsEvalKey
    );


    /**
     * Generate share of the final relinearization key.
     * @param relinearizationKey1 the intermediate relinearization key
     * @return final relinearization key share
     */
    EvalKey<DCRTPoly> generateRelinearizationKey2(
        EvalKey<DCRTPoly> relinearizationKey1
    );


    /**
     * Insert the final relinearization key in the local crypto context.
     * @param relinearizationKey the final relinearization key
     */
    void insertRelinearizationKey(
        EvalKey<DCRTPoly> relinearizationKey
    );


    /**
     * Generate share of the rotation keys.
     * @param zeroEvalRotateKeys dummy rotation keys
     * @param indices vector of rotation indices
     * @return public key share
     */
    std::shared_ptr<std::map<usint, EvalKey<DCRTPoly>>> generateRotationKeysShare(
        const std::shared_ptr<std::map<usint, EvalKey<DCRTPoly>>> zeroEvalRotateKeys,
        const std::vector<int32_t> &indices
    );


    /**
     * Insert the rotation keys in the local crypto context.
     * @param rotationKeys the roation keys
     */
    void insertRotationKeys(
        std::shared_ptr<std::map<usint, EvalKey<DCRTPoly>>> rotationKeys
    );


    /**
     * Compute the decryption share for the given ciphertext.
     * @param ciphertext the ciphertext
     * @return the corresponding decryption share
     */
    Ciphertext<DCRTPoly> computeDecryptionShare(
        const Ciphertext<DCRTPoly> &ciphertext
    );


    /**
     * Compute the bootstrapping shares for the given ciphertext.
     * @param ciphertext1 the second component of the ciphertext
     * @param commonPolynomial the common polynomial
     * @return the corresponding bootstrapping shares
     */
    std::vector<Ciphertext<DCRTPoly>> computeBootstrappingShare(
        const Ciphertext<DCRTPoly> &ciphertext1,
        Ciphertext<DCRTPoly> commonPolynomial
    );

};



namespace lbcrypto::Serial {

    template <typename T>
    std::string serialize(
        const T& obj,
        const bool verbose=false
    )
    {

        std::stringstream objSerialized;
        Serial::Serialize(obj, objSerialized, SerType::BINARY);
        std::string message = objSerialized.str();

        if (verbose)
            std::cout << "Object serialized - size: " << message.length()
                    << std::endl;
        
        return message;

    }

}



/**
 * Setup the parameters and generate the corresponding CryptoContext.
 * @param integralPrecision the bit precision for the integral part
 * @param decimalPrecision the bit precision for the decimal part, that is also
 * the (approximate) bit precision of the scaling factor
 * @param multiplicativeDepth the maximum multiplication depth circuit you can
 * evaluate, not considering the leveled needed for the bootstrapping
 * @param batchSize the number of slots used
 * @param verbose whether you want the parameters to be printed out
 * @param verboseLog pointer to string to return the verbose log, even if
 * verbose is set to false
 * @return the CryptoContext
 */
CryptoContext<DCRTPoly> generateCryptoContext(
    const usint integralPrecision,
    const usint decimalPrecision,
    const usint multiplicativeDepth,
    const usint batchSize,
    const bool verbose = true,
    std::string *verboseLog = NULL
);


/**
 * Aggregate a list of public key shares into the master public key.
 * @param cryptoContext the crypto context
 * @param publicKeyShares vector of public key shares
 * @param commonPolynomial the common polynomial
 * @return the public key (aggregation of all the public key shares)
 */
PublicKey<DCRTPoly> aggregatePublicKeyShares(
    const CryptoContext<DCRTPoly> &cryptoContext,
    const std::vector<DCRTPoly> &publicKeyShares,
    const DCRTPoly commonPolynomial
);


/**
 * Decrypt a ciphertext.
 * @param cryptoContext the crypto context
 * @param parties the parties involved
 * @param ciphertext the ciphertext
 * @param length desired plaintext length
 * @return the corresponding message
 */
std::vector<double> decrypt(
    const CryptoContext<DCRTPoly> &cryptoContext,
    const std::vector<Party> &parties,
    const Ciphertext<DCRTPoly> &ciphertext,
    const size_t length
);


/**
 * Decrypt and decode an encrypted matrix.
 * @param cryptoContext the crypto context
 * @param parties the parties involved
 * @param matrixC the encrypted matrix
 * @param ogNumRows original number of rows of the matrix
 * @param ogNumCols original number of columns of the matrix
 * @param packing true for column-wise, false for row-wise
 * @param numRowsPrevMatrix only needed for row-wise packing
 * @return the plaintext matrix
 */
std::vector<std::vector<double>> decryptMatrix(
    const CryptoContext<DCRTPoly> &cryptoContext,
    const std::vector<Party> &parties,
    const Ciphertext<DCRTPoly> &matrixC,
    const size_t ogNumRows,
    const size_t ogNumCols,
    const bool packing = true,
    const size_t numRowsPrevMatrix = 0
);


/**
 * Decrypt and decode an encrypted vector.
 * @param cryptoContext the crypto context
 * @param parties the parties involved
 * @param vectorC the encrypted vector
 * @param ogLength original length of the vector
 * @param packing true for column-wise, false for row-wise
 * @param numRowsPrevMatrix only needed for row-wise packing
 * @return the plaintext matrix
 */
std::vector<double> decryptVector(
    const CryptoContext<DCRTPoly> &cryptoContext,
    const std::vector<Party> &parties,
    const Ciphertext<DCRTPoly> &vectorC,
    const size_t ogLength,
    const bool packing,
    const size_t numRowsPrevMatrix = 0
);


/**
 * Decrypt a ciphertext and print the result.
 * @param cryptoContext the crypto context
 * @param parties the parties involved
 * @param ciphertext the ciphertext
 * @param length desired plaintext length
 * @param label description label
 * @param threshold round to zero all values below the threshold
 * @param expectedResult the expected decryption result
 * @param numCols number of columns for vertically packed vectors
 */
void printDecryption(
    const CryptoContext<DCRTPoly> &cryptoContext,
    const std::vector<Party> &parties,
    const Ciphertext<DCRTPoly> &ciphertext,
    const size_t length = 0,
    const std::string label = "",
    const double threshold = 0.0,
    const std::vector<double> expectedResult = {},
    const size_t numCols = 0
);


// Functions to simulate multiparty protocols are defined here.
namespace protocol
{

    /**
     * Setup protocol: setup the parameters and generate the corresponding
     * CryptoContext.
     * @param integralPrecision the bit precision for the integral part
     * @param decimalPrecision the bit precision for the decimal part, that is
     * also the (approximate) bit precision of the scaling factor
     * @param multiplicativeDepth the maximum multiplication depth circuit you
     * can evaluate, not considering the leveled needed for the bootstrapping
     * @param batchSize the number of slots used
     * @param parties the parties involved
     * @param verbose
     * @return the crypto context
     */
    CryptoContext<DCRTPoly> setup(
        const usint integralPrecision,
        const usint decimalPrecision,
        const usint multiplicativeDepth,
        const usint batchSize,
        std::vector<Party> &parties,
        const bool verbose
    );


    /**
     * Key generation protocol.
     * @param cryptoContext the crypto context
     * @param parties the parties involved
     * @param verbose
     * @return the public key
     */
    PublicKey<DCRTPoly> keyGeneration(
        const CryptoContext<DCRTPoly> cryptoContext,
        std::vector<Party> &parties,
        const bool verbose = true
    );


    /**
     * Relinearization key generation protocol: generate the key used to
     * evaluate multiplication gates under encryption.
     * @param cryptoContext the crypto context
     * @param parties the parties involved
     * @param verbose
     */
    void relinearizationKeyGeneration(
        const CryptoContext<DCRTPoly> cryptoContext,
        std::vector<Party> &parties,
        const bool verbose = true
    );


    /**
     * Rotation key generation protocol: generate the keys used to evaluate a
     * vector rotation under encryption.
     * @param cryptoContext the crypto context
     * @param parties the parties involved
     * @param indices vector of rotation indices
     * @param verbose
     */
    void rotationKeyGeneration(
        const CryptoContext<DCRTPoly> cryptoContext,
        std::vector<Party> &parties,
        const std::vector<int32_t> &indices,
        const bool verbose = true
    );


    /**
     * Decryption protocol.
     * @param cryptoContext the crypto context
     * @param parties the parties involved
     * @param ciphertext the ciphertext
     * @return the corresponding plaintext
     */
    Plaintext decrypt(
        const CryptoContext<DCRTPoly> &cryptoContext,
        const std::vector<Party> &parties,
        const Ciphertext<DCRTPoly> &ciphertext
    );


    /**
     * Bootstrapping protocol.
     * @param cryptoContext the crypto context
     * @param publicKey the public key
     * @param parties the parties involved
     * @param ciphertext the ciphertext
     * @return the refreshed ciphertext
     */
    Ciphertext<DCRTPoly> bootstrap(
        const CryptoContext<DCRTPoly> &cryptoContext,
        const PublicKey<DCRTPoly> &publicKey,
        const std::vector<Party> &parties,
        Ciphertext<DCRTPoly> ciphertext
    );

} // namespace protocol
