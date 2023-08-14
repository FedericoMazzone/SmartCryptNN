#include "misc.h"


DCRTPoly generateRandomPolynomial(
    const CryptoContext<DCRTPoly> &cryptoContext,
    const std::shared_ptr<DCRTPoly::Params> &params
)
{

    DCRTPoly::DugType dug;
    DCRTPoly randPoly(dug, params, Format::EVALUATION);

    return randPoly;

}


DCRTPoly generateRandomPolynomialPK(
    const CryptoContext<DCRTPoly> &cryptoContext
)
{

    const auto cryptoParams = std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(cryptoContext->GetCryptoParameters());
    const auto paramsPK = cryptoParams->GetParamsPK();

    return generateRandomPolynomial(cryptoContext, paramsPK);

}


DCRTPoly generateRandomPolynomialQP(
    const CryptoContext<DCRTPoly> &cryptoContext
)
{
    const auto cryptoParams = std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(cryptoContext->GetCryptoParameters());
    const auto paramsQP = cryptoParams->GetParamsQP();

    return generateRandomPolynomial(cryptoContext, paramsQP);

}


EvalKey<DCRTPoly> aToEvalKey(
    const CryptoContext<DCRTPoly> &cryptoContext,
    const DCRTPoly &a
)
{

    const auto cryptoParams = std::dynamic_pointer_cast<CryptoParametersRNS>(cryptoContext->GetCryptoParameters());

    const auto numPartQ = cryptoParams->GetNumPartQ();
    std::vector<DCRTPoly> av(numPartQ);
    for (usint part = 0; part < numPartQ; part++)
        av[part] = a;

    EvalKeyRelin<DCRTPoly> ek(std::make_shared<EvalKeyRelinImpl<DCRTPoly>>(cryptoContext));
    ek->SetAVector(std::move(av));

    return ek;

}


EvalKey<DCRTPoly> a0ToEvalKey(
    const CryptoContext<DCRTPoly> &cryptoContext,
    const DCRTPoly &a
)
{

    const auto cryptoParams = std::dynamic_pointer_cast<CryptoParametersRNS>(cryptoContext->GetCryptoParameters());
    const auto paramsQP = cryptoParams->GetParamsQP();

    DCRTPoly zero = DCRTPoly(paramsQP, Format::EVALUATION, true);

    const auto numPartQ = cryptoParams->GetNumPartQ();
    std::vector<DCRTPoly> av(numPartQ);
    std::vector<DCRTPoly> bv(numPartQ);
    for (usint part = 0; part < numPartQ; part++)
    {
        av[part] = a;
        bv[part] = zero;
    }

    EvalKeyRelin<DCRTPoly> ek(std::make_shared<EvalKeyRelinImpl<DCRTPoly>>(cryptoContext));
    ek->SetAVector(std::move(av));
    ek->SetBVector(std::move(bv));

    return ek;

}


EvalKey<DCRTPoly> generateZeroEvalKey(
    const CryptoContext<DCRTPoly> &cryptoContext
)
{

    const auto cryptoParams = std::dynamic_pointer_cast<CryptoParametersRNS>(cryptoContext->GetCryptoParameters());
    const auto paramsQP = cryptoParams->GetParamsQP();

    DCRTPoly zeroPoly = DCRTPoly(paramsQP, Format::EVALUATION, true);

    const auto numPartQ = cryptoParams->GetNumPartQ();
    std::vector<DCRTPoly> av(numPartQ);
    std::vector<DCRTPoly> bv(numPartQ);
    for (usint part = 0; part < numPartQ; part++)
    {
        av[part] = zeroPoly;
        bv[part] = zeroPoly;
    }

    EvalKeyRelin<DCRTPoly> ek(std::make_shared<EvalKeyRelinImpl<DCRTPoly>>(cryptoContext));
    ek->SetAVector(std::move(av));
    ek->SetBVector(std::move(bv));

    return ek;

}


PrivateKey<DCRTPoly> generateZeroPrivateKey(
    const CryptoContext<DCRTPoly> &cryptoContext
)
{

    const auto cryptoParams = std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(cryptoContext->GetCryptoParameters());
    const auto elementParams = cryptoParams->GetElementParams();

    DCRTPoly zeroPoly = DCRTPoly(elementParams, Format::EVALUATION, true);

    PrivateKey<DCRTPoly> zeroKey = std::make_shared<PrivateKeyImpl<DCRTPoly>>(cryptoContext);
    zeroKey->SetPrivateElement(std::move(zeroPoly));

    return zeroKey;

}


std::shared_ptr<std::map<usint, EvalKey<DCRTPoly>>> generateZeroRotateKeys(
    const CryptoContext<DCRTPoly> &cryptoContext,
    const std::vector<int32_t> &indices
)
{

    PrivateKey<DCRTPoly> zeroKey = generateZeroPrivateKey(cryptoContext);

    cryptoContext->EvalRotateKeyGen(zeroKey, indices);

    std::shared_ptr<std::map<usint, EvalKey<DCRTPoly>>> zeroEvalRotateKeys =
        std::make_shared<std::map<usint, EvalKey<DCRTPoly>>>(
            cryptoContext->GetEvalAutomorphismKeyMap(
                zeroKey->GetKeyTag()
            )
        );
    
    return zeroEvalRotateKeys;

}
