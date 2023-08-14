#include <iostream>
#include <string.h>
#include <filesystem>
#include <iomanip>

// header files for socket connections
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>

// header files for serialization
#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"
#include "key/key-ser.h"
#include "scheme/ckksrns/ckksrns-ser.h"


enum LogLevel {
    DEBUG,
    INFO,
    WARNING,
    ERROR
};


class Host {

    private:

    std::string name_;
    LogLevel logLevel_ = INFO;
    std::ofstream logFile_;


    public:

    Host(
        std::string name = ""
    ) : name_(name) {}


    ~Host()
    {

        if (this->logFile_.is_open())
        {
            log(LogLevel::INFO, "Closing log file.");
            this->logFile_.close();
        }

    }


    void setLogLevel(
        const LogLevel level
    )
    {

        this->logLevel_ = level;

    }


    void setLogFile(
        const std::string &filename
    )
    {

        this->logFile_.open(filename);
        // std::filesystem::permissions(
        //     filename,
        //     std::filesystem::perms::owner_read |
        //     std::filesystem::perms::owner_write |
        //     std::filesystem::perms::group_read |
        //     std::filesystem::perms::group_write |
        //     std::filesystem::perms::others_read |
        //     std::filesystem::perms::others_write
        // );

    }


    std::string getTimeStamp()
    {

        auto now = std::chrono::system_clock::now();
        std::time_t t = std::chrono::system_clock::to_time_t(now);

        // Extract milliseconds
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

        std::stringstream ss;
        ss << std::put_time(std::localtime(&t), "%F %T") << '.' << std::setfill('0') << std::setw(3) << ms.count();
        return ss.str();

    }


    template<typename... Args>
    void log(
        const LogLevel level,
        Args const &... args
    )
    {

        if (level < this->logLevel_)
            return;
        
        std::string prefix;
        switch (level) {
            case DEBUG:
                prefix = "[DEBUG] ";
                break;
            case INFO:
                prefix = "[INFO] ";
                break;
            case WARNING:
                prefix = "[WARNING] ";
                break;
            case ERROR:
                prefix = "[ERROR] ";
                break;
            default:
                break;
        }

        std::ostringstream logMessage;
        logMessage << "(" << getTimeStamp() << ") " << prefix << name_ << ": ";
        (logMessage << ... << args);
        logMessage << std::endl;

        if (this->logFile_.is_open())
        {
            this->logFile_ << logMessage.str();
            this->logFile_.flush();
        }
        else
            std::cout << logMessage.str();

    }


    void sendMsg
    (
        const int socket,
        const std::string &message
    )
    {

        ssize_t bytesSent;
        const char *messageStr = message.c_str();
        const size_t messageLength = message.length();

        log(DEBUG, "Start sending message to socket ", socket);

        // send the message length
        bytesSent = send(socket, &messageLength, sizeof(size_t), 0);
        if (bytesSent == -1) {
            log(ERROR, "Error: Failed to send message length");
            close(socket);
            return;
        }
        log(DEBUG, "Sent message length ", messageLength, " to socket ", socket);

        // send the message in chunks
        size_t bytesSentSoFar = 0;
        while (bytesSentSoFar < messageLength)
        {
            bytesSent = send(socket, messageStr + bytesSentSoFar, messageLength - bytesSentSoFar, 0);
            if (bytesSent == -1) {
                log(ERROR, "Error: Failed to send message chunk");
                close(socket);
                return;
            }
            bytesSentSoFar += bytesSent;
            // log(DEBUG, "Sent ", bytesSentSoFar, " / ", messageLength, " bytes");
        }

        // // send the message in chunks
        // size_t bytesSentSoFar = 0;
        // while (bytesSentSoFar < messageLength)
        // {
        //     bytesSent = send(socket, messageStr + bytesSentSoFar, std::min((size_t) BUFF_LENGTH, messageLength - bytesSentSoFar), 0);
        //     if (bytesSent == -1) {
        //         log(ERROR, "Error: Failed to send message chunk");
        //         close(socket);
        //         return;
        //     }
        //     bytesSentSoFar += bytesSent;
        // }

        // log(DEBUG, "Sent message of size ", bytesSentSoFar);

        log(DEBUG, "Finished sending message to socket ", socket);

    }


    std::stringstream recvMsg
    (
        const int socket
    )
    {

        ssize_t bytesReceived;
        size_t messageLength;
        std::stringstream ss;

        log(DEBUG, "Start receiving message from socket ", socket);

        // receive the message length
        bytesReceived = recv(socket, &messageLength, sizeof(size_t), 0);
        if (bytesReceived == -1) {
            log(ERROR, "Error: Failed to receive message length");
            close(socket);
            return ss;
        }
        log(DEBUG, "Received message length ", messageLength, " from socket ", socket);

        // receive the message
        char* buffer = new char[messageLength];
        size_t bytesReceivedSoFar = 0;
        while (bytesReceivedSoFar < messageLength)
        {
            bytesReceived = recv(socket, buffer + bytesReceivedSoFar, messageLength - bytesReceivedSoFar, 0);
            if (bytesReceived == -1) {
                log(ERROR, "Error: Failed to receive message chunk");
                close(socket);
                return ss;
            }
            bytesReceivedSoFar += bytesReceived;
            // log(DEBUG, "Received ", bytesReceivedSoFar, " / ", messageLength, " bytes");
        }
        ss.write(buffer, messageLength);
        delete[] buffer;

        // // receive the message in chunks
        // size_t bytesReceivedSoFar = 0;
        // while (bytesReceivedSoFar < messageLength)
        // {
        //     bytesReceived = recv(socket, buffer, std::min((size_t) BUFF_LENGTH, messageLength - bytesReceivedSoFar), 0);
        //     if (bytesReceived == -1) {
        //         log(ERROR, "Error: Failed to receive message chunk");
        //         close(socket);
        //         return ss;
        //     }
        //     ss.write(buffer, bytesReceived);
        //     bytesReceivedSoFar += bytesReceived;
        // }

        // log(DEBUG, "Received message of size ", bytesReceivedSoFar);

        log(DEBUG, "Finished receiving message from socket ", socket);

        return ss;

    }

    
    void executeProtocol(
        const std::string &protocolName,
        std::function<void()> function
    )
    {

        // common code before
        log(INFO, protocolName + ": [STARTED]");
        auto start = std::chrono::high_resolution_clock::now();

        // call the actual function
        function();

        // common code after
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        log(INFO, protocolName + ": [COMPLETED] in ", std::fixed, std::setprecision(3), elapsed_seconds.count(), "s");
        
    }


    template<typename ReturnType>
    ReturnType executeProtocol(
        const std::string &protocolName,
        std::function<ReturnType()> function
    )
    {
        
        // common code before
        log(INFO, protocolName + ": [STARTED]");
        auto start = std::chrono::high_resolution_clock::now();

        // call the actual function
        ReturnType result = function();

        // common code after
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        log(INFO, protocolName + ": [COMPLETED] in ", std::fixed, std::setprecision(3), elapsed_seconds.count(), "s");

        // return the function output
        return result;
        
    }

};
