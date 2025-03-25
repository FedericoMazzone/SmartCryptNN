#include "mp_host.h"

#include "multiparty.h"
#include "syntactic_sugar.h"
#include "../vector_utils/vector_utils.h"
#include "../data_utils/load_mnist.h"
#include "../ml_under_he/ml_under_he.h"
// #include "../he_utils/eval_logistic.h"
#include "../he_utils/matrix_multiplication.h"

#include <cassert>
#include <cstdlib>
#include <random>



#define PORT 8080

class Client : public Host, public Party {

    public:

    int clientSocket;
    size_t numClients;
    std::vector<std::vector<double>> trainX, trainY, testX, testY;
    NeuralNetwork* model0;
    EncryptedNeuralNetwork* model;
    bool isPlaintextFL;


    Client
    (
        const usint id,
        LogLevel logLevel = INFO,
        const std::string &logFile = ""
    ) : Host("client" + std::to_string(id)), Party(id)
    {

        Host::setLogLevel(logLevel);
        if (!logFile.empty())
            Host::setLogFile(logFile);
        
    }


    void startConnection
    (
        const char *serverIP
    )
    {

        // create socket
        this->clientSocket = socket(AF_INET, SOCK_STREAM, 0);
        if (this->clientSocket == -1) {
            log(ERROR, "Error: Failed to create socket");
            return;
        }

        // server address
        struct sockaddr_in serverAddress;
        memset(&serverAddress, 0, sizeof(serverAddress));
        serverAddress.sin_family = AF_INET;
        serverAddress.sin_port = htons(8080);
        serverAddress.sin_addr.s_addr = inet_addr(serverIP);

        // connect to server
        if (connect(this->clientSocket, (struct sockaddr *)&serverAddress, sizeof(serverAddress)) < 0) {
            log(ERROR, "Error: Failed to connect to server");
            return;
        }
        log(DEBUG, "Connected to server");

        // receive the number of clients from server
        std::stringstream numClientsSerialized = recvMsg();
        Serial::Deserialize(this->numClients, numClientsSerialized, SerType::BINARY);
        log(INFO, "Number of clients: ", this->numClients);

    }


    void closeConnection()
    {

        close(this->clientSocket);

    }


    void sendMsg
    (
        const std::string &message
    )
    {

        Host::sendMsg(
            this->clientSocket,
            message
        );

    }


    std::stringstream recvMsg()
    {

        return Host::recvMsg(
            this->clientSocket
        );

    }


    void setup_()
    {

        std::stringstream cryptoContextSerialized = recvMsg();

        importCryptoContext(cryptoContextSerialized);
        
    }


    void setup()
    {

        Host::executeProtocol(
            "Setup protocol",
            [
                this
            ]() {
                return this->setup_();
            }
        );

    }


    void keyGeneration_()
    {

        // generate private key share
        generatePrivateKeyShare();

        // receive common polynomial from server
        DCRTPoly commonPolynomial;
        std::stringstream commonPolynomialSerialized = recvMsg();
        Serial::Deserialize(commonPolynomial, commonPolynomialSerialized, SerType::BINARY);

        // generate the public key share
        DCRTPoly publicKeyShare = generatePublicKeyShare(commonPolynomial);

        // send the public key share to server
        std::string publicKeyShareSerialized = Serial::serialize(publicKeyShare);
        sendMsg(publicKeyShareSerialized);

        // receive public key from server
        std::stringstream publicKeySerialized = recvMsg();
        Serial::Deserialize(this->publicKey, publicKeySerialized, SerType::BINARY);
        
    }


    void keyGeneration()
    {

        Host::executeProtocol(
            "Public / Private key generation protocol",
            [
                this
            ]() {
                return this->keyGeneration_();
            }
        );

    }


    void relinearizationKeyGeneration_()
    {

        // receive commonPolynomialAsKey from server
        EvalKey<DCRTPoly> commonPolynomialAsKey;
        std::stringstream commonPolynomialAsKeySerialized = recvMsg();
        Serial::Deserialize(commonPolynomialAsKey, commonPolynomialAsKeySerialized, SerType::BINARY);

        // generate key1 share
        EvalKey<DCRTPoly> key1Share = generateRelinearizationKey1(
            commonPolynomialAsKey
        );

        // send key1Share to server
        std::string key1ShareSerialized = Serial::serialize(key1Share);
        sendMsg(key1ShareSerialized);

        // receive key1 from server
        EvalKey<DCRTPoly> key1;
        std::stringstream key1Serialized = recvMsg();
        Serial::Deserialize(key1, key1Serialized, SerType::BINARY);

        // generate key2 share
        EvalKey<DCRTPoly> key2Share = generateRelinearizationKey2(
            key1
        );

        // send key2Share to server
        std::string key2ShareSerialized = Serial::serialize(key2Share);
        sendMsg(key2ShareSerialized);

        // receive key2 from server
        EvalKey<DCRTPoly> key2;
        std::stringstream key2Serialized = recvMsg();
        Serial::Deserialize(key2, key2Serialized, SerType::BINARY);
        insertRelinearizationKey(key2);

    }


    void relinearizationKeyGeneration()
    {

        Host::executeProtocol(
            "Relinearization key generation protocol",
            [
                this
            ]() {
                return this->relinearizationKeyGeneration_();
            }
        );

    }


    void rotationKeyGeneration_(
        const std::vector<int32_t> &indices
    )
    {

        // receive zeroEvalRotateKeys from server
        std::shared_ptr<std::map<usint, EvalKey<DCRTPoly>>> zeroEvalRotateKeys;
        std::stringstream zeroEvalRotateKeysSerialized = recvMsg();
        Serial::Deserialize(zeroEvalRotateKeys, zeroEvalRotateKeysSerialized, SerType::BINARY);

        // generate rotation keys shares
        std::shared_ptr<std::map<usint, EvalKey<DCRTPoly>>> evalRotateKeysShare = generateRotationKeysShare(
            zeroEvalRotateKeys,
            indices
        );

        // send evalRotateKeysShare to server
        std::string evalRotateKeysShareSerialized = Serial::serialize(evalRotateKeysShare);
        sendMsg(evalRotateKeysShareSerialized);

        // receive evalRotateKeys from server
        std::shared_ptr<std::map<usint, EvalKey<DCRTPoly>>> evalRotateKeys;
        std::stringstream evalRotateKeysSerialized = recvMsg();
        Serial::Deserialize(evalRotateKeys, evalRotateKeysSerialized, SerType::BINARY);
        insertRotationKeys(evalRotateKeys);
        
    }


    void rotationKeyGeneration(
        const std::vector<int32_t> &indices
    )
    {

        Host::executeProtocol(
            "Rotation keys generation protocol",
            [
                this,
                indices
            ]() {
                return this->rotationKeyGeneration_(
                    indices
                );
            }
        );

    }


    Plaintext decrypt_(
        const Ciphertext<DCRTPoly> &ciphertext
    )
    {

        // send ciphertext to server
        std::string ciphertextSerialized = Serial::serialize(ciphertext);
        sendMsg(ciphertextSerialized);

        // receive other ciphertexts from server
        std::vector<Ciphertext<DCRTPoly>> ciphertexts;
        for (size_t i = 0; i < this->numClients; i++)
            if (i != this->id)
            {
                Ciphertext<DCRTPoly> receivedCiphertext;
                std::stringstream receivedCiphertextSerialized = recvMsg();
                Serial::Deserialize(receivedCiphertext, receivedCiphertextSerialized, SerType::BINARY);
                ciphertexts.push_back(receivedCiphertext);
            }
            else
                ciphertexts.push_back(ciphertext);
        
        // compute decryption shares
        std::vector<Ciphertext<DCRTPoly>> decryptionSharesOthers;
        for (Ciphertext<DCRTPoly> c : ciphertexts)
        {
            Ciphertext<DCRTPoly> decryptionShare = computeDecryptionShare(c);
            decryptionSharesOthers.push_back(decryptionShare);
        }

        // send other ciphertexts decryption shares to server
        for (size_t i = 0; i < this->numClients; i++)
            if (i != this->id)
            {
                std::string decryptionShareSerialized = Serial::serialize(decryptionSharesOthers[i]);
                sendMsg(decryptionShareSerialized);
            }

        // receive the decryption shares from server
        std::vector<Ciphertext<DCRTPoly>> decryptionShares;
        decryptionShares.push_back(ciphertext); // c0
        decryptionShares.push_back(decryptionSharesOthers[this->id]);
        for (size_t i = 0; i < this->numClients; i++)
            if (i != this->id)
            {
                Ciphertext<DCRTPoly> decryptionShare;
                std::stringstream decryptionShareSerialized = recvMsg();
                Serial::Deserialize(decryptionShare, decryptionShareSerialized, SerType::BINARY);
                decryptionShares.push_back(decryptionShare);
            }

        // aggregate the decryption shares into the resulting plaintext
        Plaintext result;
        this->cryptoContext->MultipartyDecryptFusion(decryptionShares, &result);

        return result;

    }


    Plaintext decrypt(
        const Ciphertext<DCRTPoly> &ciphertext
    )
    {

        return Host::executeProtocol<Plaintext>(
            "Decryption protocol",
            [
                this,
                ciphertext
            ]() {
                return this->decrypt_(
                    ciphertext
                );
            }
        );

    }


    
    void centralDecrypt_()
    {

        // receive ciphertext from server
        Ciphertext<DCRTPoly> ciphertext;
        std::stringstream ciphertextSerialized = recvMsg();
        Serial::Deserialize(ciphertext, ciphertextSerialized, SerType::BINARY);

        // compute decryption share
        Ciphertext<DCRTPoly> decryptionShare = computeDecryptionShare(ciphertext);

        // send decryption share to server
        std::string decryptionShareSerialized = Serial::serialize(decryptionShare);
        sendMsg(decryptionShareSerialized);

    }


    void centralDecrypt()
    {

        Host::executeProtocol(
            "Central decryption protocol",
            [
                this
            ]() {
                return this->centralDecrypt_();
            }
        );

    }


    Ciphertext<DCRTPoly> bootstrap_(
        Ciphertext<DCRTPoly> ciphertext
    )
    {

        // Content omitted until OpenFHE threshold-CKKS bootstrapping is publicly available.

    }


    Ciphertext<DCRTPoly> bootstrap(
        Ciphertext<DCRTPoly> ciphertext
    )
    {

        return Host::executeProtocol<Ciphertext<DCRTPoly>>(
            "Bootstrapping protocol",
            [
                this,
                ciphertext
            ]() {
                return this->bootstrap_(
                    ciphertext
                );
            }
        );

    }


    void centralBootstrap_()
    {

        // Content omitted until OpenFHE threshold-CKKS bootstrapping is publicly available.

    }


    void centralBootstrap()
    {

        return Host::executeProtocol(
            "Central bootstrapping protocol",
            [
                this
            ]() {
                return this->centralBootstrap_();
            }
        );

    }


    void test_()
    {

        // help decryption
        centralDecrypt();
        centralDecrypt();
        centralDecrypt();
        centralDecrypt();
        centralDecrypt();
        centralDecrypt();
        centralDecrypt();
        centralDecrypt();
        centralDecrypt();
        // centralDecrypt();
        centralBootstrap();
        centralDecrypt();
        centralDecrypt();

        // receive encrypted message from server
        Ciphertext<DCRTPoly> messageC;
        std::stringstream messageCSerialized = recvMsg();
        Serial::Deserialize(messageC, messageCSerialized, SerType::BINARY);

        // modify the message
        Ciphertext<DCRTPoly> modifiedMessageC;
        switch (id)
        {
        case 0:
            modifiedMessageC = messageC + messageC;
            break;
        
        case 1:
            modifiedMessageC = messageC >> 2;
            break;
        
        case 2:
            modifiedMessageC = messageC * messageC * messageC;
            break;
        
        default:
            break;
        }

        // bootstrap the modified encrypted message
        log(INFO, "Modified message level: ", modifiedMessageC->GetLevel());
        modifiedMessageC = bootstrap(modifiedMessageC);
        log(INFO, "Modified message level: ", modifiedMessageC->GetLevel());

        // decrypt the modified encrypted message
        Plaintext modifiedMessageP = decrypt(modifiedMessageC);
        log(INFO, "Modified message: ", modifiedMessageP);

        // send the modified encrypted message to server
        std::string modifiedMessageCSerialized = Serial::serialize(modifiedMessageC);
        sendMsg(modifiedMessageCSerialized);

    }


    void test()
    {

        Host::executeProtocol(
            "Test",
            [
                this
            ]() {
                return this->test_();
            }
        );

    }


    void loadData_(
        const size_t trainSize,
        const size_t testSize
    )
    {

        loadMNIST(
            this->trainX, this->trainY,
            this->testX, this->testY,
            trainSize, testSize,
            this->id * trainSize, this->id * testSize
        );

    }


    void loadData(
        const size_t trainSize,
        const size_t testSize
    )
    {

        Host::executeProtocol(
            "Load data",
            [
                this,
                trainSize,
                testSize
            ]() {
                return this->loadData_(
                    trainSize,
                    testSize
                );
            }
        );

    }

    
    void initializeDummyModel_(
        const std::vector<size_t> &architecture,
        const WeightInitializer weightInitializer,
        const double sigmoidBound,
        const uint32_t sigmoidDegree,
        const size_t encryptedLayerThreshold = 0
    )
    {

        assert(encryptedLayerThreshold < architecture.size());

        std::vector<size_t> architecture0(
            architecture.begin(),
            architecture.begin() + encryptedLayerThreshold + 1
        );
        std::vector<size_t> architecture1(
            architecture.begin() + encryptedLayerThreshold,
            architecture.end()
        );
        log(INFO, "Architecture (plaintext part): ", vec2str(architecture0));
        log(INFO, "Architecture (encrypted part): ", vec2str(architecture1));

        if (architecture1.size() == 1)
            this->isPlaintextFL = true;
        else
            this->isPlaintextFL = false;

        this->model0 = new NeuralNetwork(
            architecture0
        );

        if (!this->isPlaintextFL)
            this->model = new EncryptedNeuralNetwork(
                this->cryptoContext,
                this->publicKey,
                architecture1,
                -sigmoidBound,
                sigmoidBound,
                sigmoidDegree,
                weightInitializer
            );

    }


    void initializeDummyModel(
        const std::vector<size_t> &architecture,
        const WeightInitializer weightInitializer,
        const double sigmoidBound,
        const uint32_t sigmoidDegree,
        const size_t encryptedLayerThreshold = 0
    )
    {

        Host::executeProtocol(
            "Generate model",
            [
                this,
                architecture,
                weightInitializer,
                sigmoidBound,
                sigmoidDegree,
                encryptedLayerThreshold
            ]() {
                return this->initializeDummyModel_(
                    architecture,
                    weightInitializer,
                    sigmoidBound,
                    sigmoidDegree,
                    encryptedLayerThreshold
                );
            }
        );

    }


    void sendModel_()
    {

        const size_t modelDepth0 = this->model0->depth;
        const size_t modelDepth = this->isPlaintextFL ? 0 : this->model->getDepth();

        std::vector<std::string> weightsSerialized;
        std::vector<std::string> biasesSerialized;
        for (size_t i = 0; i < modelDepth0; i++)
        {
            weightsSerialized.emplace_back(
                Serial::serialize(this->model0->weights[i])
            );
            biasesSerialized.emplace_back(
                Serial::serialize(this->model0->biases[i])
            );
        }
        for (size_t i = 0; i < modelDepth; i++)
        {
            weightsSerialized.emplace_back(
                Serial::serialize(this->model->weights[i])
            );
            biasesSerialized.emplace_back(
                Serial::serialize(this->model->biases[i])
            );
        }

        for (size_t i = 0; i < weightsSerialized.size(); i++)
        {
            sendMsg(weightsSerialized[i]);
            sendMsg(biasesSerialized[i]);
        }

        for (size_t i = 0; i < modelDepth; i++)
        {
            this->centralBootstrap();
            this->centralBootstrap();
        }

    }


    void sendModel()
    {

        Host::executeProtocol(
            "Send model",
            [
                this
            ]() {
                return this->sendModel_();
            }
        );

    }


    void receiveModel_()
    {

        const size_t modelDepth0 = this->model0->depth;

        std::vector<std::vector<std::vector<double>>> weights0(modelDepth0);
        std::vector<std::vector<double>> biases0(modelDepth0);
        for (size_t i = 0; i < modelDepth0; i++)
        {
            std::stringstream weightSerialized = recvMsg();
            Serial::Deserialize(weights0[i], weightSerialized, SerType::BINARY);
            std::stringstream biasSerialized = recvMsg();
            Serial::Deserialize(biases0[i], biasSerialized, SerType::BINARY);
        }
        this->model0->weights = weights0;
        this->model0->biases = biases0;

        if (!this->isPlaintextFL)
        {
            const size_t modelDepth = this->model->getDepth();

            std::vector<Ciphertext<DCRTPoly>> weights(modelDepth);
            std::vector<Ciphertext<DCRTPoly>> biases(modelDepth);
            for (size_t i = 0; i < modelDepth; i++)
            {
                std::stringstream weightSerialized = recvMsg();
                Serial::Deserialize(weights[i], weightSerialized, SerType::BINARY);
                std::stringstream biasSerialized = recvMsg();
                Serial::Deserialize(biases[i], biasSerialized, SerType::BINARY);
            }
            this->model->weights = weights;
            this->model->biases = biases;
        }
        
    }


    void receiveModel()
    {

        Host::executeProtocol(
            "Receive model",
            [
                this
            ]() {
                return this->receiveModel_();
            }
        );

    }


    void predict_()
    {

        if (!this->isPlaintextFL)
            for (size_t i = 0; i < this->model->getDepth(); i++)
            {
                this->centralBootstrap();
                // this->centralDecrypt();
                // this->centralDecrypt();
                if (i < this->model->getDepth() - 1)
                    this->centralBootstrap();
            }

    }


    void predict()
    {

        Host::executeProtocol(
            "Predict",
            [
                this
            ]() {
                return this->predict_();
            }
        );

    }


    void evaluate_(
        const size_t testSize
    )
    {

        for (size_t id = 0; id < testSize; id++)
        {
            log(INFO, "Evaluate ", id + 1, " / ", testSize);
            this->predict();
            if (!this->isPlaintextFL)
                this->centralDecrypt();
        }

    }


    void evaluate(
        const size_t testSize
    )
    {

        Host::executeProtocol(
            "Evaluate",
            [
                this,
                testSize
            ]() {
                return this->evaluate_(
                    testSize
                );
            }
        );

    }


    Ciphertext<DCRTPoly> vectorMatrixMultPackCCProtocol(
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

        std::string protocolName = "vectorMatrixMultPackCCProtocol size " + std::to_string(numRows) + "x" + std::to_string(numCols);

        return Host::executeProtocol<Ciphertext<DCRTPoly>>(
            protocolName,
            [
                this,
                cryptoContext,
                publicKey,
                vectorC,
                matrixC,
                ogNumCols,
                numRows,
                numCols,
                packing,
                masking,
                transposing
            ]() {
                return vectorMatrixMultPackCC(
                    cryptoContext,
                    publicKey,
                    vectorC,
                    matrixC,
                    ogNumCols,
                    numRows,
                    numCols,
                    packing,
                    masking,
                    transposing
                );
            }
        );

    }


    Ciphertext<DCRTPoly> evalLogisticProtocol(
        Ciphertext<DCRTPoly> ciphertext,
        double a,
        double b,
        uint32_t degree
    )
    {

        std::string protocolName = "evalLogisticProtocol degree " + std::to_string(degree);

        return Host::executeProtocol<Ciphertext<DCRTPoly>>(
            protocolName,
            [
                this,
                ciphertext,
                a,
                b,
                degree
            ]() {
                return this->cryptoContext->EvalLogistic(
                    ciphertext,
                    a,
                    b,
                    degree
                );
            }
        );

    }


    Ciphertext<DCRTPoly> evalLogisticDerivativeProtocol(
        Ciphertext<DCRTPoly> ciphertext,
        double a,
        double b,
        uint32_t degree
    )
    {

        std::string protocolName = "evalLogisticDerivativeProtocol degree " + std::to_string(degree);

        return Host::executeProtocol<Ciphertext<DCRTPoly>>(
            protocolName,
            [
                this,
                ciphertext,
                a,
                b,
                degree
            ]() {
                return this->cryptoContext->EvalChebyshevFunction(
                    [](double x) -> double { return std::exp(-x) / ((1 + std::exp(-x)) * (1 + std::exp(-x))); },
                    ciphertext,
                    a,
                    b,
                    degree
                );
            }
        );

    }


    void oneTrainPass_(
        const std::vector<double> &feature,
        const std::vector<double> &label,
        std::vector<Ciphertext<DCRTPoly>> *gw,
        std::vector<Ciphertext<DCRTPoly>> *gb,
        std::vector<std::vector<std::vector<double>>> *gw0,
        std::vector<std::vector<double>> *gb0
    )
    {

        const size_t modelDepth0 = this->model0->depth;
        const size_t modelDepth = this->isPlaintextFL ? 0 : this->model->getDepth();

        std::vector<std::vector<double>> l0(modelDepth0 + 1);
        std::vector<std::vector<double>> u0(modelDepth0);
        std::vector<std::vector<double>> e0(modelDepth0);

        std::vector<Ciphertext<DCRTPoly>> l(modelDepth + 1);
        std::vector<Ciphertext<DCRTPoly>> u(modelDepth);
        std::vector<Ciphertext<DCRTPoly>> e(modelDepth);
        
        // feedforward in plaintext
        l0[0] = feature;
        for (size_t layerID = 0; layerID < modelDepth0; layerID++)
        {
            u0[layerID] = vectorMatrixMult(l0[layerID], this->model0->weights[layerID]);
            u0[layerID] = addVectors(u0[layerID], this->model0->biases[layerID]);
            l0[layerID + 1] = sigmoid(u0[layerID]);
        }
        
        if (!this->isPlaintextFL)
        {
            // feedforward in ciphertext
            // to improve: allow for plaintext vector
            l[0] = this->cryptoContext->Encrypt(
                this->publicKey,
                this->cryptoContext->MakeCKKSPackedPlaintext(l0[modelDepth0])
            );
            for (size_t i = 0; i < modelDepth; i++)
            {
                const std::string protocolName = "Encrypted layer forward";
                log(INFO, protocolName + ": [STARTED]");
                auto start = std::chrono::high_resolution_clock::now();

                u[i] = this->vectorMatrixMultPackCCProtocol(
                    this->cryptoContext, this->publicKey,
                    l[i], this->model->weights[i],
                    this->model->architecture[i + 1],
                    this->model->rows[i], this->model->cols[i],
                    this->model->packing[i],
                    true, false
                );
                u[i] = u[i] + this->model->biases[i];
                u[i] = this->bootstrap(u[i]);

                l[i + 1] = this->evalLogisticProtocol(
                    u[i],
                    this->model->sigmoidLeftBound, this->model->sigmoidRightBound,
                    this->model->sigmoidDegree
                );
                l[i + 1] = this->bootstrap(l[i + 1]);
                l[i + 1] = maskVector(
                    this->cryptoContext,
                    l[i + 1],
                    !this->model->packing[i],
                    this->model->architecture[i + 1],
                    this->model->rows[i], this->model->cols[i]
                );
                l[i + 1] = this->bootstrap(l[i + 1]);

                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed_seconds = end - start;
                log(INFO, protocolName + ": [COMPLETED] in ", std::fixed, std::setprecision(3), elapsed_seconds.count(), "s");
            }

            // backpropagation in ciphertext
            Ciphertext<DCRTPoly> gwtmp, logisticDerU;
            for (size_t i = modelDepth; i-- > 0; )
            {
                const std::string protocolName = "Encrypted layer backward";
                log(INFO, protocolName + ": [STARTED]");
                auto start = std::chrono::high_resolution_clock::now();

                if (i == modelDepth - 1)
                {
                    std::vector<double> y;
                    if (!this->model->packing[i])
                        y = label;
                    else
                    {
                        size_t rows, cols;
                        y = encodeMatrix(
                            transpose({label}),
                            rows, cols,
                            false, this->model->rows[i]
                        );
                    }
                    e[i] = y - l[modelDepth];
                }
                else
                {
                    e[i] = this->vectorMatrixMultPackCCProtocol(
                        this->cryptoContext,
                        this->publicKey,
                        e[i + 1], this->model->weights[i + 1],
                        this->model->architecture[i + 1],
                        this->model->cols[i + 1], this->model->rows[i + 1],
                        !this->model->packing[i + 1],
                        false, false
                    );
                }

                logisticDerU = this->evalLogisticDerivativeProtocol(
                    u[i],
                    this->model->sigmoidLeftBound, this->model->sigmoidRightBound,
                    this->model->sigmoidDegree
                );
                logisticDerU = this->bootstrap(logisticDerU);
                log(INFO, "logisticDerU ", i, " level: ", logisticDerU->GetLevel());

                e[i] = e[i] * logisticDerU;
                log(INFO, "e * logisticDerU ", i, " level: ", e[i]->GetLevel());
                e[i] = maskVector(
                    this->cryptoContext,
                    e[i],
                    !this->model->packing[i],
                    this->model->architecture[i + 1],
                    this->model->rows[i], this->model->cols[i]
                );
                log(INFO, "e * logisticDerU * mask ", i, " level: ", e[i]->GetLevel());
                e[i] = this->bootstrap(e[i]);

                (*gb)[i] = (*gb)[i] + e[i];

                gwtmp = vectorTransposeVectorMultCC(
                    this->cryptoContext,
                    l[i], e[i],
                    this->model->rows[i], this->model->cols[i],
                    this->model->packing[i]
                );
                (*gw)[i] = (*gw)[i] + gwtmp;

                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed_seconds = end - start;
                log(INFO, protocolName + ": [COMPLETED] in ", std::fixed, std::setprecision(3), elapsed_seconds.count(), "s");
            }

            if (modelDepth0 > 0)
            {
                Ciphertext<DCRTPoly> eIntermediate = this->vectorMatrixMultPackCCProtocol(
                    this->cryptoContext,
                    this->publicKey,
                    e[0], this->model->weights[0],
                    this->model->architecture[0],
                    this->model->cols[0], this->model->rows[0],
                    !this->model->packing[0],
                    false, false
                );

                assert(u0[modelDepth0 - 1].size() == this->model->architecture[0]);
                assert(this->model->packing[0] == true);

                Plaintext eIntermediateP = this->decrypt(eIntermediate);
                e0[modelDepth0 - 1] = decodeVector(
                    eIntermediateP,
                    this->model->architecture[0],
                    true
                );

            }
        }
        else
            e0[modelDepth0 - 1] = addVectors(
                label,
                multVectors(
                    -1,
                    l0[modelDepth0]
                )
            );

        // backpropagation in plaintext
        for (size_t layerID = modelDepth0; layerID-- > 0; )
        {
            if (layerID < modelDepth0 - 1)
                e0[layerID] = vectorMatrixMult(
                    e0[layerID + 1],
                    transpose(this->model0->weights[layerID + 1])
                );
            e0[layerID] = multVectors(
                sigmoidDerivative(u0[layerID]),
                e0[layerID]
            );
            (*gb0)[layerID] = addVectors(
                (*gb0)[layerID],
                e0[layerID]
            );
            (*gw0)[layerID] = addMatrices(
                (*gw0)[layerID],
                matrixMult(
                    transpose({l0[layerID]}),
                    {e0[layerID]}
                )
            );
        }

    }


    void oneTrainPass(
        const std::vector<double> &feature,
        const std::vector<double> &label,
        std::vector<Ciphertext<DCRTPoly>> *gw,
        std::vector<Ciphertext<DCRTPoly>> *gb,
        std::vector<std::vector<std::vector<double>>> *gw0,
        std::vector<std::vector<double>> *gb0
    )
    {

        Host::executeProtocol(
            "One train pass",
            [
                this,
                feature,
                label,
                gw,
                gb,
                gw0,
                gb0
            ]() {
                return this->oneTrainPass_(
                    feature,
                    label,
                    gw,
                    gb,
                    gw0,
                    gb0
                );
            }
        );

    }


    void train_(
        const std::vector<std::vector<double>> &trainFeatures,
        const std::vector<std::vector<double>> &trainLabels,
        const size_t numEpochs,
        double learningRate,
        size_t batchSize = 0,
        const double l2coeff = 0.0,
        const size_t testSize = 0
    )
    {

        assert(trainFeatures.size() == trainLabels.size());

        if (batchSize == 0) batchSize = trainFeatures.size();

        log(INFO, "Train size        : ", trainFeatures.size());
        log(INFO, "Test size         : ", testSize);
        log(INFO, "Batch size        : ", batchSize);
        log(INFO, "Epochs            : ", numEpochs);
        log(INFO, "Learning rate     : ", learningRate);
        log(INFO, "L2 coefficient    : ", l2coeff);

        const size_t modelDepth0 = this->model0->depth;
        const size_t modelDepth = this->isPlaintextFL ? 0 : this->model->getDepth();

        // for (size_t i = 0; i < modelDepth; i++)
        //     this->centralDecrypt();

        Plaintext zeroP;
        
        if (!this->isPlaintextFL)
            zeroP = this->cryptoContext->MakeCKKSPackedPlaintext(
                std::vector<double>{}
            );
        std::vector<Ciphertext<DCRTPoly>> gw;
        std::vector<Ciphertext<DCRTPoly>> gb;

        std::vector<std::vector<std::vector<double>>> gw0 = this->model0->weights;
        std::vector<std::vector<double>> gb0 = this->model0->biases;

        for (size_t epoch = 0; epoch < numEpochs; epoch++)
        {
            log(INFO, "[Epoch ", epoch + 1, "]");

            for (size_t batch = 0; batch < 1.0 * trainFeatures.size() / batchSize; batch++)
            {
                log(INFO, "batch ", batch + 1);

                // prepare batch
                const std::vector<std::vector<double>> batchX(
                    trainFeatures.begin() + batch * batchSize,
                    min(
                        trainFeatures.begin() + (batch + 1) * batchSize,
                        trainFeatures.end()
                    )
                );
                const std::vector<std::vector<double>> batchY(
                    trainLabels.begin() + batch * batchSize,
                    min(
                        trainLabels.begin() + (batch + 1) * batchSize,
                        trainLabels.end()
                    )
                );

                // reset gradients
                gw.clear();
                gb.clear();
                for (size_t i = 0; i < modelDepth; i++)
                {
                    gw.push_back(this->cryptoContext->Encrypt(this->publicKey, zeroP));
                    gb.push_back(this->cryptoContext->Encrypt(this->publicKey, zeroP));
                }

                for (auto &v1 : gw0) for (auto &v2 : v1) std::fill(v2.begin(), v2.end(), 0.0);
                for (auto &v : gb0) std::fill(v.begin(), v.end(), 0.0);

                this->receiveModel();

                // update gradients
                for (size_t id = 0; id < batchX.size(); id++)
                    this->oneTrainPass(
                        batchX[id],
                        batchY[id],
                        &gw,
                        &gb,
                        &gw0,
                        &gb0
                    );

                log(INFO, "Update model parameters: [STARTED]");
                // update model parameters
                for (size_t layerID = 0; layerID < modelDepth0; layerID++)
                {
                    this->model0->biases[layerID] = addVectors(
                        this->model0->biases[layerID],
                        multVectors(
                            learningRate / batchX.size(),
                            gb0[layerID]
                        )
                    );
                    this->model0->weights[layerID] = addMatrices(
                        matrixMult(
                            1.0 - learningRate * l2coeff / batchX.size(),
                            this->model0->weights[layerID]
                        ),
                        matrixMult(
                            learningRate / batchX.size(),
                            gw0[layerID]
                        )
                    );
                }
                for (size_t i = 0; i < modelDepth; i++)
                {
                    // this->model->biases[i] + (learningRate / batchX.size()) * gb[i];
                    this->model->biases[i] = this->model->biases[i] + (learningRate / batchX.size()) * gb[i];
                    log(INFO, "b", i, " =  update [level ", this->model->biases[i]->GetLevel(), "]");

                    if (l2coeff != 0.0)
                        // (1.0 - learningRate * l2coeff / batchX.size()) * this->model->weights[i];
                        this->model->weights[i] = (1.0 - learningRate * l2coeff / batchX.size()) * this->model->weights[i];

                    // this->model->weights[i] + (learningRate / batchX.size()) * gw[i];
                    this->model->weights[i] = this->model->weights[i] + (learningRate / batchX.size()) * gw[i];
                    log(INFO, "w", i, " =  update [level ", this->model->weights[i]->GetLevel(), "]");
                }
                log(INFO, "Update model parameters: [COMPLETED]");

                this->sendModel();
            }

            // for (size_t i = 0; i < modelDepth; i++)
            //     this->centralDecrypt();

            if (testSize > 0)
                this->evaluate(
                    testSize
                );

        }

    }


    void train(
        const std::vector<std::vector<double>> &trainFeatures,
        const std::vector<std::vector<double>> &trainLabels,
        const size_t numEpochs,
        double learningRate,
        size_t batchSize = 0,
        const double l2coeff = 0.0,
        const size_t testSize = 0
    )
    {

        Host::executeProtocol(
            "Training",
            [
                this,
                trainFeatures,
                trainLabels,
                numEpochs,
                learningRate,
                batchSize,
                l2coeff,
                testSize
            ]() {
                return this->train_(
                    trainFeatures,
                    trainLabels,
                    numEpochs,
                    learningRate,
                    batchSize,
                    l2coeff,
                    testSize
                );
            }
        );

    }

};



int main(int argc, char const *argv[]) {

    const size_t encryptedLayerThreshold = 0;

    // model parameters
    std::vector<size_t> architecture = {64, 30, 20, 10};
    WeightInitializer weightInitializer = WeightInitializer::UNIFORM;
    double sigmoidBound = 10.0;
    uint32_t sigmoidDegree = 14;

    // Training Parameters
    const double learningRate = 3.0;
    const size_t numEpochs = 300;
    const size_t trainSize = 30;
    const size_t testSize = 100;
    const size_t batchSize = 10;
    const double l2coeff = 0.005;

    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <server IP address> <client ID>" << std::endl;
        return -1;
    }
    const usint id = atoi(argv[2]);
    const LogLevel logLevel = DEBUG;
    const std::string logFile = "client" + std::to_string(id) + ".log";

    usint usedSlots = 2048;
    std::vector<int32_t> rotationIndices = {};
    for (int i = 0; i <= log2(usedSlots); i++)
    {
        rotationIndices.push_back(1 << i);
        rotationIndices.push_back(-(1 << i));
    }

    std::random_device rd;
    std::srand(rd());

    Client client(
        id,
        logLevel,
        logFile
    );

    client.startConnection(argv[1]);

    
    if (encryptedLayerThreshold < architecture.size() - 1)
    {
        client.setup();

        client.keyGeneration();

        client.relinearizationKeyGeneration();

        client.rotationKeyGeneration(
            rotationIndices
        );
    }

    // client.test();

    client.loadData(
        trainSize,
        0
    );

    client.initializeDummyModel(
        architecture,
        weightInitializer,
        sigmoidBound,
        sigmoidDegree,
        encryptedLayerThreshold
    );

    client.train(
        client.trainX,
        client.trainY,
        numEpochs,
        learningRate,
        batchSize,
        l2coeff,
        testSize
    );

    // std::vector<Ciphertext<DCRTPoly>> gw;
    // std::vector<Ciphertext<DCRTPoly>> gb;
    // const Plaintext zeroP = client.cryptoContext->MakeCKKSPackedPlaintext(
    //     std::vector<double>{}
    // );
    // for (size_t i = 0; i < client.model->getDepth(); i++)
    // {
    //     gw.push_back(client.cryptoContext->Encrypt(client.publicKey, zeroP));
    //     gb.push_back(client.cryptoContext->Encrypt(client.publicKey, zeroP));
    // }
    // client.oneTrainPass(
    //     client.trainX[0],
    //     client.trainY[0],
    //     &gw,
    //     &gb
    // );

    client.closeConnection();

    return 0;

}
