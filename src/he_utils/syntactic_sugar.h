#include "openfhe.h"

using namespace lbcrypto;


/**
 * Proposal for ciphertext shifting and plaintext-ciphertext addition operators.
 * They should go inside CiphertextImpl class.
*/
template <typename Element>
Ciphertext<Element> operator<<(const Ciphertext<Element>& a, int32_t index) {
    return a->GetCryptoContext()->EvalRotate(a, index);
}

template <typename Element>
Ciphertext<Element> operator>>(const Ciphertext<Element>& a, int32_t index) {
    return a->GetCryptoContext()->EvalRotate(a, -index);
}

template <typename Element>
Ciphertext<Element> operator+(const Ciphertext<Element>& a, const Plaintext &b) {
    return a->GetCryptoContext()->EvalAdd(a, b);
}

template <typename Element>
Ciphertext<Element> operator+(const Plaintext &a, const Ciphertext<Element>& b) {
    return b + a;
}

template <typename Element>
Ciphertext<Element> operator+(const Ciphertext<Element>& a, const std::vector<double> &b) {
    CryptoContext<Element> cc = a->GetCryptoContext();
    // if (cc->getSchemeId() != "CKKSRNS") {
    //     // throw suitable exception
    // }
    auto plaintext = cc->MakeCKKSPackedPlaintext(b);
    return a + plaintext;
}

template <typename Element>
Ciphertext<Element> operator+(const std::vector<double> &a, const Ciphertext<Element>& b) {
    return b + a;
}

template <typename Element>
Ciphertext<Element> operator-(const Ciphertext<Element>& a, const std::vector<double> &b) {
    std::vector<double> bNeg(b.size());
    for (size_t i = 0; i < b.size(); i++)
        bNeg[i] = -b[i];
    return a + bNeg;
}

template <typename Element>
Ciphertext<Element> operator-(const std::vector<double> &a, const Ciphertext<Element>& b) {
    return a + (-b);
}

template <typename Element>
Ciphertext<Element> operator+(const Ciphertext<Element>& a, const double b) {
    return a->GetCryptoContext()->EvalAdd(a, b);
}

template <typename Element>
Ciphertext<Element> operator+(const double a, const Ciphertext<Element>& b) {
    return b + a;
}

template <typename Element>
Ciphertext<Element> operator*(const Ciphertext<Element>& a, const double b) {
    return a->GetCryptoContext()->EvalMult(a, b);
}

template <typename Element>
Ciphertext<Element> operator*(const double a, const Ciphertext<Element>& b) {
    return b * a;
}

template <typename Element>
Ciphertext<Element> operator*(const Ciphertext<Element>& a, const std::vector<double> &b) {
    CryptoContext<Element> cc = a->GetCryptoContext();
    auto plaintext = cc->MakeCKKSPackedPlaintext(b);
    return cc->EvalMult(a, plaintext);
}

template <typename Element>
Ciphertext<Element> operator*(const std::vector<double> &a, const Ciphertext<Element>& b) {
    return b * a;
}
