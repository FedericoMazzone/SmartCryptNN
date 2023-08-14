#include "mp_host.h"

#include "multiparty.h"
#include "syntactic_sugar.h"
#include "misc.h"
#include "../data_utils/load_mnist.h"
#include "../ml_under_he/ml_under_he.h"
#include "../he_utils/matrix_multiplication.h"
#include "../vector_utils/vector_utils.h"

#include <cassert>
#include <random>



#define PORT 8080
#define NUM_CLIENTS 3

class Server : public Host {

    public:

    int serverSocket;
    std::vector<int> clientSockets;
    CryptoContext<DCRTPoly> cryptoContext;
    PublicKey<DCRTPoly> publicKey;
    std::vector<std::vector<double>> trainX, trainY, testX, testY;
    NeuralNetwork* model0;
    EncryptedNeuralNetwork* model;
    bool isPlaintextFL;


    Server(
        LogLevel logLevel = INFO,
        const std::string &logFile = ""
    ) : Host("server")
    {
        
        Host::setLogLevel(logLevel);
        if (!logFile.empty())
            Host::setLogFile(logFile);
        
    }


    void startConnection()
    {

        // create a server socket
        this->serverSocket = socket(AF_INET, SOCK_STREAM, 0);
        if (this->serverSocket < 0) {
            log(ERROR, "Error: Failed to create socket");
            return;
        }

        // configure the server socket
        struct sockaddr_in serverAddress;
        serverAddress.sin_family = AF_INET;
        serverAddress.sin_addr.s_addr = INADDR_ANY;
        serverAddress.sin_port = htons(PORT);
        int bind_result = bind(this->serverSocket, (struct sockaddr *)&serverAddress, sizeof(serverAddress));
        if (bind_result < 0) {
            log(ERROR, "Error: Failed to bind socket");
            return;
        }

        // listen for incoming connections
        int listenResult = listen(this->serverSocket, NUM_CLIENTS);
        if (listenResult < 0) {
            log(ERROR, "Error: Failed to listen on socket");
            return;
        }

        // accept incoming connections in a loop
        while (this->clientSockets.size() < NUM_CLIENTS) {

            // accept a new connection
            struct sockaddr_in clientAddress;
            socklen_t clientAddressSize = sizeof(clientAddress);
            int clientSocket = accept(this->serverSocket, (struct sockaddr *)&clientAddress, &clientAddressSize);
            if (clientSocket < 0) {
                log(ERROR, "Error: Failed to accept client connection");
                continue;
            }

            this->clientSockets.push_back(clientSocket);
            log(DEBUG, "New client connected: ", this->clientSockets.size(), " / ", NUM_CLIENTS);

        }

        // send the number of clients to clients
        std::string numClientsSerialized = Serial::serialize((size_t) NUM_CLIENTS);
        sendMsgAsync(numClientsSerialized);

    }


    void closeConnection()
    {

        for (int clientSocket : this->clientSockets)
            close(clientSocket);
        close(this->serverSocket);

    }


    void sendMsgAsync
    (
        const std::vector<std::string>& messages
    )
    {

        assert(messages.size() == this->clientSockets.size());

        std::vector<std::thread> threads;
        for (size_t i = 0; i < this->clientSockets.size(); i++)
            threads.emplace_back([this, i, &messages]() {
                sendMsg(this->clientSockets[i], messages[i]);
            });

        for (auto& thread : threads)
            thread.join();

    }


    void sendMsgAsync
    (
        const std::string& message
    )
    {

        const std::vector<std::string> messages(
            this->clientSockets.size(),
            message
        );

        sendMsgAsync(messages);

    }


    std::vector<std::stringstream> recvMsgAsync()
    {

        std::vector<std::stringstream> messages(this->clientSockets.size());

        std::vector<std::thread> threads;
        for (size_t i = 0; i < this->clientSockets.size(); i++)
            threads.emplace_back([this, i, &messages]() {
                messages[i] = recvMsg(this->clientSockets[i]);
            });
        
        for (auto& thread : threads)
            thread.join();

        return messages;

    }


    void setup_
    (
        const usint integralPrecision,
        const usint decimalPrecision,
        const usint multiplicativeDepth,
        const usint batchSize
    )
    {

        std::string verboseLog;

        this->cryptoContext = generateCryptoContext(
            integralPrecision,
            decimalPrecision,
            multiplicativeDepth,
            batchSize,
            false,
            &verboseLog
        );

        log(INFO, verboseLog);

        // send the cryptoContext to clients
        std::string cryptoContextSerialized = Serial::serialize(this->cryptoContext);
        sendMsgAsync(cryptoContextSerialized);

    }


    void setup
    (
        const usint integralPrecision,
        const usint decimalPrecision,
        const usint multiplicativeDepth,
        const usint batchSize
    )
    {

        Host::executeProtocol(
            "Setup protocol",
            [
                this,
                integralPrecision,
                decimalPrecision,
                multiplicativeDepth,
                batchSize
            ]() {
                return this->setup_(
                    integralPrecision,
                    decimalPrecision,
                    multiplicativeDepth,
                    batchSize
                );
            }
        );

    }


    void keyGeneration_()
    {

        // generate common polynomial
        DCRTPoly commonPolynomial = generateRandomPolynomialPK(this->cryptoContext);

        // send the commonPolynomial to clients
        std::string commonPolynomialSerialized = Serial::serialize(commonPolynomial);
        sendMsgAsync(commonPolynomialSerialized);

        // receive publicKeyShares from clients
        std::vector<DCRTPoly> publicKeyShares(this->clientSockets.size());
        std::vector<std::thread> threads;
        for (size_t i = 0; i < this->clientSockets.size(); i++)
            threads.emplace_back([this, i, &publicKeyShares]() {
                std::stringstream publicKeyShareSerialized = recvMsg(this->clientSockets[i]);
                Serial::Deserialize(publicKeyShares[i], publicKeyShareSerialized, SerType::BINARY);
            });
        for (auto& thread : threads)
            thread.join();

        // aggregate the public key shares into the master public key
        this->publicKey = aggregatePublicKeyShares(
            this->cryptoContext,
            publicKeyShares,
            commonPolynomial
        );

        // send public key to clients
        std::string publicKeySerialized = Serial::serialize(this->publicKey);
        sendMsgAsync(publicKeySerialized);

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

        // generate common polynomial
        DCRTPoly commonPolynomial = generateRandomPolynomialQP(this->cryptoContext);
        EvalKey<DCRTPoly> commonPolynomialAsKey = aToEvalKey(
            this->cryptoContext,
            commonPolynomial
        );

        // send the commonPolynomialAsKey to clients
        std::string commonPolynomialAsKeySerialized = Serial::serialize(commonPolynomialAsKey);
        sendMsgAsync(commonPolynomialAsKeySerialized);

        // receive key1Shares from clients
        std::vector<EvalKey<DCRTPoly>> key1Shares(this->clientSockets.size());
        std::vector<std::thread> threads;
        for (size_t i = 0; i < this->clientSockets.size(); i++)
            threads.emplace_back([this, i, &key1Shares]() {
                std::stringstream key1ShareSerialized = recvMsg(this->clientSockets[i]);
                Serial::Deserialize(key1Shares[i], key1ShareSerialized, SerType::BINARY);
            });
        for (auto& thread : threads)
            thread.join();

        // aggregate key1Shares into key1
        EvalKey<DCRTPoly> key1 = a0ToEvalKey(
            this->cryptoContext,
            commonPolynomial
        );
        for (EvalKey<DCRTPoly> &key1Share : key1Shares)
            key1 = this->cryptoContext->MultiAddEvalKeys(
                key1,
                key1Share
            );

        // send key1 to clients
        std::string key1Serialized = Serial::serialize(key1);
        sendMsgAsync(key1Serialized);

        // receive key2Shares from clients
        std::vector<EvalKey<DCRTPoly>> key2Shares(this->clientSockets.size());
        threads.clear();
        for (size_t i = 0; i < this->clientSockets.size(); i++)
            threads.emplace_back([this, i, &key2Shares]() {
                std::stringstream key2ShareSerialized = recvMsg(this->clientSockets[i]);
                Serial::Deserialize(key2Shares[i], key2ShareSerialized, SerType::BINARY);
            });
        for (auto& thread : threads)
            thread.join();

        // aggregate key2Shares into key2
        EvalKey<DCRTPoly> key2 = generateZeroEvalKey(
            this->cryptoContext
        );
        for (EvalKey<DCRTPoly> &key2Share : key2Shares)
        {
            key2 = this->cryptoContext->MultiAddEvalMultKeys(
                key2,
                key2Share
            );
        }

        // send key2 to clients
        std::string key2Serialized = Serial::serialize(key2);
        sendMsgAsync(key2Serialized);

        cryptoContext->InsertEvalMultKey({key2});

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

        // generate zeroEvalRotateKeys
        std::shared_ptr<std::map<usint, EvalKey<DCRTPoly>>> zeroEvalRotateKeys =
            generateZeroRotateKeys(
                cryptoContext,
                indices
            );
        
        // send zeroEvalRotateKeys to clients
        std::string zeroEvalRotateKeysSerialized = Serial::serialize(zeroEvalRotateKeys);
        sendMsgAsync(zeroEvalRotateKeysSerialized);

        // receive evalRotateKeysShares from clients
        std::vector<std::shared_ptr<std::map<usint, EvalKey<DCRTPoly>>>> evalRotateKeysShares(this->clientSockets.size());
        std::vector<std::thread> threads;
        for (size_t i = 0; i < this->clientSockets.size(); i++)
            threads.emplace_back([this, i, &evalRotateKeysShares]() {
                std::stringstream evalRotateKeysShareSerialized = recvMsg(this->clientSockets[i]);
                Serial::Deserialize(evalRotateKeysShares[i], evalRotateKeysShareSerialized, SerType::BINARY);
            });
        for (auto& thread : threads)
            thread.join();

        // aggregate evalRotateKeysShares into evalRotateKeys
        std::shared_ptr<std::map<usint, EvalKey<DCRTPoly>>> evalRotateKeys =
            zeroEvalRotateKeys;
        for (std::shared_ptr<std::map<usint, EvalKey<DCRTPoly>>> &evalRotateKeysShare : evalRotateKeysShares)
        {
            evalRotateKeys = cryptoContext->MultiAddEvalAutomorphismKeys(
                evalRotateKeys,
                evalRotateKeysShare
            );
        }

        // send evalRotateKeys to clients
        std::string evalRotateKeysSerialized = Serial::serialize(evalRotateKeys);
        sendMsgAsync(evalRotateKeysSerialized);

        cryptoContext->InsertEvalAutomorphismKey(evalRotateKeys);

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


    // In this draft version, the server is acting like an hub.
    // Need to allow also for direct client-client communication, but then we
    // lose coordination, so need for unstructured communication.
    // Note that in this version the server may sum the all but one decryption shares.
    void decrypt_()
    {

        std::vector<std::stringstream> ciphertextsSerialized(NUM_CLIENTS);
        std::vector<std::stringstream> decryptionSharesSerialized(NUM_CLIENTS * NUM_CLIENTS);

        std::vector<std::thread> threads;
        for (size_t i = 0; i < NUM_CLIENTS; i++)
            threads.emplace_back([this, i, &ciphertextsSerialized]() {
                // receive ciphertexts from clients
                ciphertextsSerialized[i] = recvMsg(this->clientSockets[i]);
            });
        for (auto& thread : threads)
            thread.join();
        
        threads.clear();
        for (size_t i = 0; i < NUM_CLIENTS; i++)
            threads.emplace_back([this, i, &ciphertextsSerialized, &decryptionSharesSerialized]() {
                // send ciphertexts to clients
                for (size_t j = 0; j < NUM_CLIENTS; j++)
                    if (j != i)
                        sendMsg(this->clientSockets[i], ciphertextsSerialized[j].str());
                // receive decryption shares from clients
                for (size_t j = 0; j < NUM_CLIENTS; j++)
                    if (j != i)
                        decryptionSharesSerialized[i * NUM_CLIENTS + j] = recvMsg(this->clientSockets[i]);
            });
        for (auto& thread : threads)
            thread.join();
        
        threads.clear();
        for (size_t i = 0; i < NUM_CLIENTS; i++)
            threads.emplace_back([this, i, &decryptionSharesSerialized]() {
                // redistribute the decryption shares to clients
                for (size_t j = 0; j < NUM_CLIENTS; j++)
                    if (j != i)
                        sendMsg(this->clientSockets[i], decryptionSharesSerialized[j * NUM_CLIENTS + i].str());
            });
        for (auto& thread : threads)
            thread.join();

    }


    void decrypt()
    {

        Host::executeProtocol(
            "Decryption protocol",
            [
                this
            ]() {
                return this->decrypt_();
            }
        );

    }


    Plaintext centralDecrypt_(
        const Ciphertext<DCRTPoly> &ciphertext
    )
    {

        // send ciphertext to clients
        std::string ciphertextSerialized = Serial::serialize(ciphertext);
        sendMsgAsync(ciphertextSerialized);

        // receive decryption share from clients
        std::vector<Ciphertext<DCRTPoly>> decryptionShares(NUM_CLIENTS + 1);
        decryptionShares[NUM_CLIENTS] = ciphertext; // c0
        std::vector<std::thread> threads;
        for (size_t i = 0; i < NUM_CLIENTS; i++)
            threads.emplace_back([this, i, &decryptionShares]() {
                std::stringstream decryptionSharesSerialized = recvMsg(this->clientSockets[i]);
                Serial::Deserialize(decryptionShares[i], decryptionSharesSerialized, SerType::BINARY);
            });
        for (auto& thread : threads)
            thread.join();

        // aggregate the decryption shares into the resulting plaintext
        Plaintext result;
        this->cryptoContext->MultipartyDecryptFusion(decryptionShares, &result);

        return result;

    }


    Plaintext centralDecrypt(
        const Ciphertext<DCRTPoly> &ciphertext
    )
    {

        return Host::executeProtocol<Plaintext>(
            "Central decryption protocol",
            [
                this,
                ciphertext
            ]() {
                return this->centralDecrypt_(
                    ciphertext
                );
            }
        );

    }


    // In this draft version, the server is acting like an hub.
    // Need to allow also for direct client-client communication, but then we
    // lose coordination, so need for unstructured communication.
    // Note that in this version the server may sum the all but one decryption shares.
    void bootstrap_()
    {
        
        // Content omitted until OpenFHE threshold-CKKS bootstrapping is publicly available.

    }


    void bootstrap()
    {

        Host::executeProtocol(
            "Bootstrapping protocol",
            [
                this
            ]() {
                return this->bootstrap_();
            }
        );

    }


    Ciphertext<DCRTPoly> centralBootstrap_(
        Ciphertext<DCRTPoly> ciphertext
    )
    {

        // Content omitted until OpenFHE threshold-CKKS bootstrapping is publicly available.

    }


    Ciphertext<DCRTPoly> centralBootstrap(
        Ciphertext<DCRTPoly> ciphertext
    )
    {

        return Host::executeProtocol<Ciphertext<DCRTPoly>>(
            "Central bootstrapping protocol",
            [
                this,
                ciphertext
            ]() {
                return this->centralBootstrap_(
                    ciphertext
                );
            }
        );

    }


    void test_()
    {

        // initialize message
        std::vector<std::complex<double>> message(
            {-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0}
        );

        // encode message
        Plaintext messageP = this->cryptoContext->MakeCKKSPackedPlaintext(message);
        log(INFO, "Original plaintext: ", messageP);

        // encrypt message
        Ciphertext<DCRTPoly> messageC = this->cryptoContext->Encrypt(this->publicKey, messageP);

        // play with the ciphertext
        Ciphertext<DCRTPoly> m2C = messageC * messageC;
        Ciphertext<DCRTPoly> m3C = m2C * messageC;
        Ciphertext<DCRTPoly> m4C = m3C * messageC;
        Ciphertext<DCRTPoly> m5C = m4C * messageC;
        Ciphertext<DCRTPoly> m6C = m5C * messageC;
        Ciphertext<DCRTPoly> m7C = m6C * messageC;
        Ciphertext<DCRTPoly> m8C = m7C * messageC;
        Ciphertext<DCRTPoly> m9C = m8C * messageC;
        // Ciphertext<DCRTPoly> m10C = m9C * messageC;
        Plaintext mP = centralDecrypt(messageC);
        Plaintext m2P = centralDecrypt(m2C);
        Plaintext m3P = centralDecrypt(m3C);
        Plaintext m4P = centralDecrypt(m4C);
        Plaintext m5P = centralDecrypt(m5C);
        Plaintext m6P = centralDecrypt(m6C);
        Plaintext m7P = centralDecrypt(m7C);
        Plaintext m8P = centralDecrypt(m8C);
        Plaintext m9P = centralDecrypt(m9C);
        // Plaintext m10P = centralDecrypt(m10C);
        mP->SetLength(messageP->GetLength());
        m2P->SetLength(messageP->GetLength());
        m3P->SetLength(messageP->GetLength());
        m4P->SetLength(messageP->GetLength());
        m5P->SetLength(messageP->GetLength());
        m6P->SetLength(messageP->GetLength());
        m7P->SetLength(messageP->GetLength());
        m8P->SetLength(messageP->GetLength());
        m9P->SetLength(messageP->GetLength());
        // m10P->SetLength(messageP->GetLength());
        log(INFO, "Decryption of m [level ", messageC->GetLevel(), "]: ", mP);
        log(INFO, "Decryption of m2 [level ", m2C->GetLevel(), "]: ", m2P);
        log(INFO, "Decryption of m3 [level ", m3C->GetLevel(), "]: ", m3P);
        log(INFO, "Decryption of m4 [level ", m4C->GetLevel(), "]: ", m4P);
        log(INFO, "Decryption of m5 [level ", m5C->GetLevel(), "]: ", m5P);
        log(INFO, "Decryption of m6 [level ", m6C->GetLevel(), "]: ", m6P);
        log(INFO, "Decryption of m7 [level ", m7C->GetLevel(), "]: ", m7P);
        log(INFO, "Decryption of m8 [level ", m8C->GetLevel(), "]: ", m8P);
        log(INFO, "Decryption of m9 [level ", m9C->GetLevel(), "]: ", m9P);
        // log(INFO, "Decryption of m10 [level ", m10C->GetLevel(), "]: ", m10P);
        Ciphertext<DCRTPoly> m6BC = centralBootstrap(m6C);
        Ciphertext<DCRTPoly> m7BC = m6BC * messageC;
        Plaintext m6BP = centralDecrypt(m6BC);
        Plaintext m7BP = centralDecrypt(m7BC);
        m6BP->SetLength(messageP->GetLength());
        m7BP->SetLength(messageP->GetLength());
        log(INFO, "Decryption of m6B [level ", m6BC->GetLevel(), "]: ", m6BP);
        log(INFO, "Decryption of m7B [level ", m7BC->GetLevel(), "]: ", m7BP);

        // send message to clients
        std::string messageCSerialized = Serial::serialize(messageC);
        sendMsgAsync(messageCSerialized);

        // help clients bootstrapping the modified encrypted messages
        bootstrap();
        
        // help clients decrypting the modified encrypted messages
        decrypt();
        
        // receive modifiedMessagesC from clients
        std::vector<Ciphertext<DCRTPoly>> modifiedMessagesC(this->clientSockets.size());
        std::vector<std::thread> threads;
        for (size_t i = 0; i < this->clientSockets.size(); i++)
            threads.emplace_back([this, i, &modifiedMessagesC]() {
                std::stringstream modifiedMessageCSerialized = recvMsg(this->clientSockets[i]);
                Serial::Deserialize(modifiedMessagesC[i], modifiedMessageCSerialized, SerType::BINARY);
            });
        for (auto& thread : threads)
            thread.join();

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
            trainSize, testSize
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


    void generateModel_(
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


    void generateModel(
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
                return this->generateModel_(
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

        std::vector<std::thread> threads;
        for (size_t i = 0; i < NUM_CLIENTS; i++)
            threads.emplace_back([this, i, &weightsSerialized, &biasesSerialized]() {
                for (size_t j = 0; j < weightsSerialized.size(); j++)
                {
                    sendMsg(this->clientSockets[i], weightsSerialized[j]);
                    sendMsg(this->clientSockets[i], biasesSerialized[j]);
                }
            });
        for (auto& thread : threads)
            thread.join();

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


    void receiveModels_()
    {

        const size_t modelDepth0 = this->model0->depth;
        const size_t modelDepth = this->isPlaintextFL ? 0 : this->model->getDepth();

        std::vector<std::vector<std::vector<double>>> weightsVec0(NUM_CLIENTS * modelDepth0);
        std::vector<std::vector<double>> biasesVec0(NUM_CLIENTS * modelDepth0);
        std::vector<Ciphertext<DCRTPoly>> weightsVec(NUM_CLIENTS * modelDepth);
        std::vector<Ciphertext<DCRTPoly>> biasesVec(NUM_CLIENTS * modelDepth);

        std::vector<std::thread> threads;
        for (size_t i = 0; i < NUM_CLIENTS; i++)
            threads.emplace_back([this, i, modelDepth, modelDepth0, &weightsVec0, &biasesVec0, &weightsVec, &biasesVec]() {
                for (size_t j = 0; j < modelDepth0; j++)
                {
                    std::stringstream weightSerialized = recvMsg(this->clientSockets[i]);
                    Serial::Deserialize(weightsVec0[i * modelDepth0 + j], weightSerialized, SerType::BINARY);
                    std::stringstream biasSerialized = recvMsg(this->clientSockets[i]);
                    Serial::Deserialize(biasesVec0[i * modelDepth0 + j], biasSerialized, SerType::BINARY);
                }
                for (size_t j = 0; j < modelDepth; j++)
                {
                    std::stringstream weightSerialized = recvMsg(this->clientSockets[i]);
                    Serial::Deserialize(weightsVec[i * modelDepth + j], weightSerialized, SerType::BINARY);
                    std::stringstream biasSerialized = recvMsg(this->clientSockets[i]);
                    Serial::Deserialize(biasesVec[i * modelDepth + j], biasSerialized, SerType::BINARY);
                }
            });
        for (auto& thread : threads)
            thread.join();
        
        log(INFO, "Aggregation: [STARTED]");

        std::vector<std::vector<std::vector<double>>> weights0(modelDepth0);
        std::vector<std::vector<double>> biases0(modelDepth0);
        std::vector<Ciphertext<DCRTPoly>> weights(modelDepth);
        std::vector<Ciphertext<DCRTPoly>> biases(modelDepth);
        for (size_t j = 0; j < modelDepth0; j++)
        {
            weights0[j] = weightsVec0[j];
            biases0[j] = biasesVec0[j];
            for (size_t i = 1; i < NUM_CLIENTS; i++)
            {
                weights0[j] = addMatrices(
                    weights0[j],
                    weightsVec0[i * modelDepth0 + j]
                );
                biases0[j] = addVectors(
                    biases0[j],
                    biasesVec0[i * modelDepth0 + j]
                );
            }
            weights0[j] = matrixMult(1.0 / NUM_CLIENTS, weights0[j]);
            biases0[j] = multVectors(1.0 / NUM_CLIENTS, biases0[j]);
        }
        for (size_t j = 0; j < modelDepth; j++)
        {
            weights[j] = weightsVec[j];
            biases[j] = biasesVec[j];
            for (size_t i = 1; i < NUM_CLIENTS; i++)
            {
                weights[j] += weightsVec[i * modelDepth + j];
                biases[j] += biasesVec[i * modelDepth + j];
            }
            weights[j] = (1.0 / NUM_CLIENTS) * weights[j];
            weights[j] = this->centralBootstrap(weights[j]);
            biases[j] = (1.0 / NUM_CLIENTS) * biases[j];
            biases[j] = this->centralBootstrap(biases[j]);
        }
        
        this->model0->weights = weights0;
        this->model0->biases = biases0;
        if (!this->isPlaintextFL)
        {
            this->model->weights = weights;
            this->model->biases = biases;
        }

        log(INFO, "Aggregation: [COMPLETED]");

    }


    void receiveModels()
    {

        Host::executeProtocol(
            "Receive models",
            [
                this
            ]() {
                return this->receiveModels_();
            }
        );

    }


    Ciphertext<DCRTPoly> predict_(
        const std::vector<double> &input
    )
    {

        std::vector<double> x = input;
        for (size_t layerID = 0; layerID < this->model0->depth; layerID++)
        {
            x = vectorMatrixMult(x, this->model0->weights[layerID]);
            x = addVectors(x, this->model0->biases[layerID]);
            x = sigmoid(x);
        }

        // To improve: allow for plaintext vector
        Plaintext xP = this->cryptoContext->MakeCKKSPackedPlaintext(x);
        Ciphertext<DCRTPoly> xC = this->cryptoContext->Encrypt(this->publicKey, xP);

        for (size_t i = 0; i < this->model->getDepth(); i++)
        {
            xC = vectorMatrixMultPackCC(
                this->cryptoContext, this->publicKey,
                xC, this->model->weights[i],
                this->model->architecture[i + 1],
                this->model->rows[i], this->model->cols[i],
                this->model->packing[i],
                false, false
            ) + this->model->biases[i];

            xC = centralBootstrap(xC);

            log(INFO, "w", i, ".level = ", this->model->weights[i]->GetLevel());
            log(INFO, "b", i, ".level = ", this->model->biases[i]->GetLevel());

            // Plaintext xP = centralDecrypt(xC);
            // std::vector<double> u = decodeVector(
            //     xP,
            //     this->model->architecture[i + 1],
            //     !this->model->packing[i],
            //     this->model->architecture[i]
            // );
            // log(INFO, "u", i, ": ", vec2str(u));

            xC = this->cryptoContext->EvalLogistic(
                xC,
                this->model->sigmoidLeftBound, this->model->sigmoidRightBound,
                this->model->sigmoidDegree
            );

            // xP = centralDecrypt(xC);
            // std::vector<double> s = decodeVector(
            //     xP,
            //     this->model->architecture[i + 1],
            //     !this->model->packing[i],
            //     this->model->architecture[i]
            // );
            // log(INFO, "s", i, ": ", vec2str(s));

            if (i < this->model->getDepth() - 1)
            {
                xC = centralBootstrap(xC);

                xC = maskVector(
                    this->cryptoContext,
                    xC,
                    !this->model->packing[i],
                    this->model->architecture[i + 1],
                    this->model->rows[i], this->model->cols[i]
                );
            }
        }

        return xC;

    }


    Ciphertext<DCRTPoly> predict(
        const std::vector<double> &input
    )
    {

        return Host::executeProtocol<Ciphertext<DCRTPoly>>(
            "Predict",
            [
                this,
                input
            ]() {
                return this->predict_(
                    input
                );
            }
        );

    }


    std::vector<double> predictOnlyPlaintext_(
        const std::vector<double> &input
    )
    {

        std::vector<double> x = input;
        for (size_t layerID = 0; layerID < this->model0->depth; layerID++)
        {
            x = vectorMatrixMult(x, this->model0->weights[layerID]);
            x = addVectors(x, this->model0->biases[layerID]);
            x = sigmoid(x);
        }

        return x;

    }


    std::vector<double> predictOnlyPlaintext(
        const std::vector<double> &input
    )
    {

        return Host::executeProtocol<std::vector<double>>(
            "Predict",
            [
                this,
                input
            ]() {
                return this->predictOnlyPlaintext_(
                    input
                );
            }
        );

    }


    double evaluate_(
        const std::vector<std::vector<double>> &features,
        const std::vector<std::vector<double>> &labels
    )
    {

        assert(features.size() == labels.size());

        uint32_t correct = 0;

        for (size_t id = 0; id < features.size(); id++)
        {
            log(INFO, "Evaluate ", id + 1, " / ", features.size());

            std::vector<double> result;

            if (this->isPlaintextFL)
                result = this->predictOnlyPlaintext(features[id]);
            else
            {
                Ciphertext<DCRTPoly> resultC = this->predict(features[id]);
                Plaintext resultP = this->centralDecrypt(resultC);
                result = decodeVector(
                    resultP,
                    this->model->getOutputSize(),
                    !this->model->packing[this->model->getDepth() - 1],
                    this->model->rows[this->model->getDepth() - 1]
                );
            }

            if (argmax(result) == argmax(labels[id]))
                correct++;
        }

        double accuracy = 1.0 * correct / features.size();

        return accuracy;

    }


    double evaluate(
        const std::vector<std::vector<double>> &features,
        const std::vector<std::vector<double>> &labels
    )
    {

        return Host::executeProtocol<double>(
            "Evaluate",
            [
                this,
                features,
                labels
            ]() {
                return this->evaluate_(
                    features,
                    labels
                );
            }
        );

    }


    void oneTrainPass_()
    {
        
        if (!this->isPlaintextFL)
        {
            // feedforward in ciphertext
            for (size_t i = 0; i < this->model->getDepth(); i++)
            {
                this->bootstrap();
                this->bootstrap();
                this->bootstrap();
            }

            // backpropagation in ciphertext
            for (size_t i = this->model->getDepth(); i-- > 0; )
            {
                this->bootstrap();
                this->bootstrap();
            }

            if (this->model0->depth > 0)
                this->decrypt();
        }

    }


    void oneTrainPass()
    {

        Host::executeProtocol(
            "One train pass",
            [
                this
            ]() {
                return this->oneTrainPass_();
            }
        );

    }


    void train_(
        const size_t numEpochs,
        const size_t trainSize,
        size_t batchSize = 0,
        const double l2coeff = 0.0,
        const std::vector<std::vector<double>> &testFeatures = std::vector<std::vector<double>>(),
        const std::vector<std::vector<double>> &testLabels = std::vector<std::vector<double>>()
    )
    {

        assert(testFeatures.size() == testLabels.size());

        if (batchSize == 0) batchSize = trainSize;

        log(INFO, "Train size        : ", trainSize);
        log(INFO, "Test size         : ", testFeatures.size());
        log(INFO, "Batch size        : ", batchSize);
        log(INFO, "Epochs            : ", numEpochs);
        log(INFO, "L2 coefficient    : ", l2coeff);

        // for (size_t i = 0; i < this->model->getDepth(); i++)
        // {
        //     Plaintext wP = centralDecrypt(this->model->weights[i]);
        //     std::vector<std::vector<double>> w = decodeMatrix(
        //         wP,
        //         this->model->architecture[i],
        //         this->model->architecture[i + 1],
        //         this->model->packing[i],
        //         this->model->packing[i] ? 0 : this->model->architecture[i - 1]
        //     );
        //     log(INFO, "w", i, ": ", mat2str(w));
        // }

        for (size_t epoch = 0; epoch < numEpochs; epoch++)
        {
            log(INFO, "[Epoch ", epoch + 1, "]");

            for (size_t batch = 0; batch < 1.0 * trainSize / batchSize; batch++)
            {
                log(INFO, "batch ", batch + 1);

                this->sendModel();

                for (size_t id = 0; id < std::min(batchSize, trainSize - batch * batchSize); id++)
                    this->oneTrainPass();

                this->receiveModels();
            }

            // for (size_t i = 0; i < this->model->getDepth(); i++)
            // {
            //     Plaintext wP = centralDecrypt(this->model->weights[i]);
            //     std::vector<std::vector<double>> w = decodeMatrix(
            //         wP,
            //         this->model->architecture[i],
            //         this->model->architecture[i + 1],
            //         this->model->packing[i],
            //         this->model->packing[i] ? 0 : this->model->architecture[i - 1]
            //     );
            //     log(INFO, "w", i, ": ", mat2str(w));
            // }

            if (testFeatures.size() > 0)
            {
                double accuracy = this->evaluate(
                    testFeatures,
                    testLabels
                );
                log(INFO, "Accuracy: ", accuracy);
            }

        }

    }


    void train(
        const size_t numEpochs,
        const size_t trainSize,
        size_t batchSize = 0,
        const double l2coeff = 0.0,
        const std::vector<std::vector<double>> &testFeatures = std::vector<std::vector<double>>(),
        const std::vector<std::vector<double>> &testLabels = std::vector<std::vector<double>>()
    )
    {

        Host::executeProtocol(
            "Training",
            [
                this,
                numEpochs,
                trainSize,
                batchSize,
                l2coeff,
                testFeatures,
                testLabels
            ]() {
                return this->train_(
                    numEpochs,
                    trainSize,
                    batchSize,
                    l2coeff,
                    testFeatures,
                    testLabels
                );
            }
        );

    }

};

int main(int argc, char const *argv[]) {

    const size_t encryptedLayerThreshold = 0;

    // logging parameters
    const LogLevel logLevel = DEBUG;
    const std::string logFile = "server.log";

    // model parameters
    std::vector<size_t> architecture = {64, 30, 20, 10};
    WeightInitializer weightInitializer = WeightInitializer::UNIFORM;
    double sigmoidBound = 10.0;
    uint32_t sigmoidDegree = 14;

    // Training Parameters
    const size_t numEpochs = 300;
    const size_t trainSize = 30;
    const size_t testSize = 100;
    const size_t batchSize = 10;
    const double l2coeff = 0.005;

    // encryption scheme parameters
    usint integralPrecision = 5;
    usint decimalPrecision = 55;
    usint multiplicativeDepth = 8;
    usint usedSlots = 2048;
    std::vector<int32_t> rotationIndices = {};
    for (int i = 0; i <= log2(usedSlots); i++)
    {
        rotationIndices.push_back(1 << i);
        rotationIndices.push_back(-(1 << i));
    }

    std::random_device rd;
    std::srand(rd());

    Server server(
        logLevel,
        logFile
    );

    server.startConnection();

    if (encryptedLayerThreshold < architecture.size() - 1)
    {
        server.setup(
            integralPrecision,
            decimalPrecision,
            multiplicativeDepth,
            usedSlots
        );

        server.keyGeneration();

        server.relinearizationKeyGeneration();

        server.rotationKeyGeneration(
            rotationIndices
        );
    }

    // server.test();

    server.loadData(
        0,
        testSize
    );

    server.generateModel(
        architecture,
        weightInitializer,
        sigmoidBound,
        sigmoidDegree,
        encryptedLayerThreshold
    );

    server.train(
        numEpochs,
        trainSize,
        batchSize,
        l2coeff,
        server.testX,
        server.testY
    );

    server.closeConnection();
    
    return 0;

}
