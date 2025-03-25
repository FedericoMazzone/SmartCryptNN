#include "openfhe.h"

using namespace lbcrypto;



/**
 * Generate polynomial uniformly at random with PK params.
 * @param cryptoContext the crypto context
 * @return random polynomial
 */
DCRTPoly generateRandomPolynomialPK(
    const CryptoContext<DCRTPoly> &cryptoContext
);


/**
 * Generate polynomial uniformly at random with QP params.
 * @param cryptoContext the crypto context
 * @return random polynomial
 */
DCRTPoly generateRandomPolynomialQP(
    const CryptoContext<DCRTPoly> &cryptoContext
);


/**
 * Turn polynomial into EvalKey as AVector, for compatibility with KeySwitchGen.
 * @param cryptoContext the crypto context
 * @param a the input polynomial
 * @return the corresponding EvalKey
 */
EvalKey<DCRTPoly> aToEvalKey(
    const CryptoContext<DCRTPoly> &cryptoContext,
    const DCRTPoly &a
);


/**
 * Turn polynomial into EvalKey as AVector with zero BVector component, for
 * compatibility with KeySwitchGen.
 * @param cryptoContext the crypto context
 * @param a the input polynomial
 * @return the corresponding EvalKey
 */
EvalKey<DCRTPoly> a0ToEvalKey(
    const CryptoContext<DCRTPoly> &cryptoContext,
    const DCRTPoly &a
);


/**
 * Generate zero EvalKey, for compatibility with KeySwitchGen.
 * @param cryptoContext the crypto context
 * @return the zero EvalKey
 */
EvalKey<DCRTPoly> generateZeroEvalKey(
    const CryptoContext<DCRTPoly> &cryptoContext
);


/**
 * Generate zero PrivateKey.
 * @param cryptoContext the crypto context
 * @return the zero PrivateKey
 */
PrivateKey<DCRTPoly> generateZeroPrivateKey(
    const CryptoContext<DCRTPoly> &cryptoContext
);


/**
 * Generate zero RotateKeys, for compatibility with MultiEvalAtIndexKeyGen.
 * @param cryptoContext the crypto context
 * @param indices vector of rotation indices
 * @return the zero RotateKeys
 */
std::shared_ptr<std::map<usint, EvalKey<DCRTPoly>>> generateZeroRotateKeys(
    const CryptoContext<DCRTPoly> &cryptoContext,
    const std::vector<int32_t> &indices
);
