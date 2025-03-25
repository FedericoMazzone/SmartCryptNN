#include <iostream>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <string.h>
#include <vector>
#include <algorithm>

using namespace std;

#define PORT 8080

int main(int argc, char const *argv[]) {
    int server_fd, new_socket, valread;
    struct sockaddr_in address;
    int opt = 1;
    int addrlen = sizeof(address);
    char buffer[1024] = {0};
    vector<int> client_sockets;

    // Creating socket file descriptor
    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        perror("socket failed");
        exit(EXIT_FAILURE);
    }

    // Forcefully attaching socket to the port 8080
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT,
                                                  &opt, sizeof(opt))) {
        perror("setsockopt");
        exit(EXIT_FAILURE);
    }
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons( PORT );

    // Forcefully attaching socket to the port 8080
    if (bind(server_fd, (struct sockaddr *)&address,
                                 sizeof(address))<0) {
        perror("bind failed");
        exit(EXIT_FAILURE);
    }
    if (listen(server_fd, 3) < 0) {
        perror("listen");
        exit(EXIT_FAILURE);
    }

    // if ((new_socket = accept(server_fd, (struct sockaddr *)&address,
    //                    (socklen_t*)&addrlen))<0) {
    //     perror("accept");
    //     exit(EXIT_FAILURE);
    // }

    // while (true) {
    //     valread = read(new_socket, buffer, 1024);

    //     if (valread == 0) {
    //         // Client disconnected
    //         break;
    //     }

    //     printf("Received message from client: %s\n", buffer);

    //     // Echo the same message back to the client
    //     send(new_socket, buffer, strlen(buffer), 0);

    //     memset(buffer, 0, sizeof(buffer));
    // }

    // close(new_socket);
    // close(server_fd);

    client_sockets.push_back(server_fd);
    printf("Waiting for connections...\n");

    while (true) {
        fd_set read_fds;
        FD_ZERO(&read_fds);
        int max_fd = 0;
        for (size_t i = 0; i < client_sockets.size(); i++) {
            int fd = client_sockets[i];
            FD_SET(fd, &read_fds);
            max_fd = max(max_fd, fd);
        }

        if (select(max_fd + 1, &read_fds, NULL, NULL, NULL) < 0) {
            perror("select");
            exit(EXIT_FAILURE);
        }

        for (size_t i = 0; i < client_sockets.size(); i++) {
            int fd = client_sockets[i];
            if (FD_ISSET(fd, &read_fds)) {
                if (fd == server_fd) {
                    // New connection
                    if ((new_socket = accept(server_fd, (struct sockaddr *)&address, (socklen_t*)&addrlen))<0) {
                        perror("accept");
                        exit(EXIT_FAILURE);
                    }
                    client_sockets.push_back(new_socket);
                    cout << "New client connected!" << endl;
                } else {
                    // Incoming data from client
                    valread = read(fd, buffer, 1024);
                    if (valread == 0) {
                        // Client disconnected
                        cout << "Client disconnected." << endl;
                        close(fd);
                        client_sockets.erase(client_sockets.begin() + i);
                    } else {
                        cout << "Received message from client: " << buffer << endl;
                        // Echo the same message back to the client
                        send(fd, buffer, strlen(buffer), 0);
                    }
                    memset(buffer, 0, sizeof(buffer));
                }
            }
        }
    }

    return 0;
}
