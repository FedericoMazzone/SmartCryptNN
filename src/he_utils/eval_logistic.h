#pragma once


#include "openfhe.h"


using namespace lbcrypto;



Ciphertext<DCRTPoly> evalLogisticDerivative(
    CryptoContext<DCRTPoly> cryptoContext,
    ConstCiphertext<DCRTPoly> ciphertext,
    double a,
    double b,
    uint32_t degree
)
{
    return cryptoContext->EvalChebyshevFunction(
        [](double x) -> double { return std::exp(-x) / ((1 + std::exp(-x)) * (1 + std::exp(-x))); },
        ciphertext,
        a,
        b,
        degree
    );
}
