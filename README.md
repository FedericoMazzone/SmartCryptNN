# SmartCryptNN

Prototype for collaborative training of a neural network under partial encryption of the model.

> **_NOTE:_**  The code will be refactored for the camera ready version.

> **_NOTE:_**  The portions of the code related to threshold bootstrapping are temporarily omitted until publicly released in the OpenFHE library.

## How to install and run this prototype

1. Download a Mininet VM at http://mininet.org/download/.
2. Follow the documentation at https://openfhe-development.readthedocs.io/en/latest/ to install OpenFHE.
3. Possibly need to export the lib path.
4. Clone this repository.
5. `mkdir -p build`
6. `cd build`
7. `cmake ..`
8. `make`
9. `cd ..`
10. Download MNIST dataset: `python3 src/data_utils/download_mnist.py`
11. `python3 src/mininet_script.py`
