#include <iostream>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <string.h>

using namespace std;

#define PORT 8080

int main(int argc, char const *argv[]) {

    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <server IP address>" << endl;
        return -1;
    }

    int sock = 0, valread;
    struct sockaddr_in serv_addr;
    char buffer[1024] = {0};
    string user_input;

    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        printf("\n Socket creation error \n");
        return -1;
    }

    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(PORT);
    // Convert IPv4 and IPv6 addresses from text to binary form
    if(inet_pton(AF_INET, argv[1], &serv_addr.sin_addr)<=0) {
        printf("\nInvalid address/ Address not supported \n");
        return -1;
    }

    if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
        printf("\nConnection Failed \n");
        return -1;
    }

    while (true) {
        cout << "Enter message to send to server: ";
        getline(cin, user_input);

        if (user_input.empty()) {
            break;
        }

        send(sock, user_input.c_str(), user_input.size(), 0);

        valread = read(sock, buffer, 1024);

        if (valread == -1) {
            printf("Error reading from server\n");
            break;
        }
        else if (valread == 0) {
            printf("Server disconnected\n");
            break;
        }

        printf("%s\n",buffer );

        memset(buffer, 0, sizeof(buffer));
    }

    close(sock);

    return 0;
}
