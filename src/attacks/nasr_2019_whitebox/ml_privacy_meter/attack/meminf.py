'''
The Attack class.
'''
import logging

import numpy as np
import tensorflow as tf

from ..utils.attack_utils import attack_utils, sanity_check
from ..utils.losses import CrossEntropyLoss, mse
from ..utils.optimizers import optimizer_op
from .meminf_modules.create_cnn import (cnn_for_cnn_gradients,
                                        cnn_for_cnn_layeroutputs,
                                        cnn_for_fcn_gradients)
from .meminf_modules.create_fcn import fcn_module
from .meminf_modules.encoder import create_encoder

# To decide what attack component (FCN or CNN) to
# use on the basis of the layer name.
# CNN_COMPONENTS_LIST are the layers requiring each input in 3 dimensions.
# GRAD_COMPONENTS_LIST are the layers which have trainable components for computing gradients
CNN_COMPONENT_LIST = ['Conv', 'MaxPool']
GRAD_LAYERS_LIST = ['Conv', 'Dense']


class initialize(object):
    """
    This attack was originally proposed by Nasr et al. It exploits
    one-hot encoding of true labels, loss value, intermediate layer
    activations and gradients of intermediate layers of the target model
    on data points, for training the attack model to infer membership
    in training data.

    Paper link: https://arxiv.org/abs/1812.00910

    Args:
    ------
    target_train_model: The target (classification) model that'll
                        be used to train the attack model.

    target_attack_model: The target (classification) model that we are
                         interested in quantifying the privacy risk of.
                         The trained attack model will be used
                         for attacking this model to quantify its membership
                         privacy leakage.

    train_datahandler: an instance of `ml_privacy_meter.data.attack_data.load`,
                       that is used to retrieve dataset for training the
                       attack model. The member set of this training set is
                       a subset of the classification model's
                       training set. Check Main README on how to
                       load dataset for the attack.

    attack_datahandler: an instance of `ml_privacy_meter.data.attack_data.load`,
                        used to retrieve dataset for evaluating the attack
                        model. The member set of this test/evaluation set is
                        a subset of the target attack model's train set minus
                        the training members of the target_train_model.

    optimizer: The optimizer op for training the attack model.
               Default op is "adam".

    layers_to_exploit: a list of integers specifying the indices of layers,
                       whose activations will be exploited by the attack model.
                       If the list has only a single element and
                       it is equal to the index of last layer,
                       the attack can be considered as a "blackbox" attack.

    gradients_to_exploit: a list of integers specifying the indices
                          of layers whose gradients will be
                          exploited by the attack model.

    exploit_loss: boolean; whether to exploit loss value of target model or not.

    exploit_label: boolean; whether to exploit one-hot encoded labels or not.

    learning_rate: learning rate for training the attack model.

    epochs: Number of epochs to train the attack model.
    """

    def __init__(self,
                 target_train_model,
                 target_attack_model,
                 train_datahandler,
                 attack_datahandler,
                 optimizer="adam",
                 model_name="sample_model",
                 layers_to_exploit=None,
                 gradients_to_exploit=None,
                 exploit_loss=True,
                 exploit_label=True,
                 learning_rate=0.001,
                 epochs=100
                 ):

        self.attack_utils = attack_utils()
        self.target_train_model = target_train_model
        self.target_attack_model = target_attack_model
        self.train_datahandler = train_datahandler
        self.attack_datahandler = attack_datahandler
        self.optimizer = optimizer_op(optimizer, learning_rate)
        self.layers_to_exploit = layers_to_exploit
        self.gradients_to_exploit = gradients_to_exploit
        self.exploit_loss = exploit_loss
        self.exploit_label = exploit_label
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.output_size = int(target_train_model.output.shape[1])
        self.ohencoding = self.attack_utils.createOHE(self.output_size)
        self.model_name = model_name

    def initialize_attack_model(self):
        """
        Initializes a `tf.keras.Model` object for attack model.
        The output of the attack is the output of the encoder module.
        """
        # Create input containers for attack & encoder model.
        self.create_input_containers()
        layers = self.target_train_model.layers

        # basic sanity checks
        sanity_check(layers, self.layers_to_exploit)
        sanity_check(layers, self.gradients_to_exploit)

        # Create individual attack components
        self.create_attack_components(layers)

        # Initialize the attack model
        output = self.encoder
        self.attackmodel = tf.keras.Model(
            inputs=self.attackinputs, outputs=output)

    def create_input_containers(self):
        """
        Creates arrays for inputs to the attack and 
        encoder model. 
        (NOTE: Although the encoder is a part of the attack model, 
        two sets of containers are required for connecting 
        the TensorFlow graph).
        """
        self.attackinputs = []
        self.encoderinputs = []

    def create_layer_components(self, layers):
        """
        Creates CNN or FCN components for layers to exploit
        """
        for l in self.layers_to_exploit:
            # For each layer to exploit, module created and added to self.attackinputs and self.encoderinputs
            layer = layers[l-1]
            input_shape = layer.output_shape[1]
            requires_cnn = map(lambda i: i in layer.__class__.__name__,
                               CNN_COMPONENT_LIST)
            if any(requires_cnn):
                module = cnn_for_cnn_layeroutputs(layer.output_shape)
            else:
                module = fcn_module(input_shape, 100)
            self.attackinputs.append(module.input)
            self.encoderinputs.append(module.output)

    def create_label_component(self, output_size):
        """
        Creates component if OHE label is to be exploited
        """
        module = fcn_module(output_size)
        self.attackinputs.append(module.input)
        self.encoderinputs.append(module.output)

    def create_loss_component(self):
        """
        Creates component if loss value is to be exploited
        """
        module = fcn_module(1, 100)
        self.attackinputs.append(module.input)
        self.encoderinputs.append(module.output)

    def create_gradient_components(self, model, layers):
        """
        Creates CNN/FCN component for gradient values of layers of gradients to exploit
        """
        grad_layers = []
        for layer in layers:
            if any(map(lambda i: i in layer.__class__.__name__, GRAD_LAYERS_LIST)):
                grad_layers.append(layer)
        variables = model.variables
        for layerindex in self.gradients_to_exploit:
            # For each gradient to exploit, module created and added to self.attackinputs and self.encoderinputs
            layer = grad_layers[layerindex-1]
            shape = self.attack_utils.get_gradshape(variables, layerindex)
            requires_cnn = map(lambda i: i in layer.__class__.__name__,
                               CNN_COMPONENT_LIST)
            if any(requires_cnn):
                module = cnn_for_cnn_gradients(shape)
            else:
                module = cnn_for_fcn_gradients(shape)
            self.attackinputs.append(module.input)
            self.encoderinputs.append(module.output)

    def create_attack_components(self, layers):
        """
        Creates FCN and CNN modules constituting the attack model.  
        """
        model = self.target_train_model

        # for layer outputs
        if self.layers_to_exploit and len(self.layers_to_exploit):
            self.create_layer_components(layers)

        # for one hot encoded labels
        if self.exploit_label:
            self.create_label_component(self.output_size)

        # for loss
        if self.exploit_loss:
            self.create_loss_component()

        # for gradients
        if self.gradients_to_exploit and len(self.gradients_to_exploit):
            self.create_gradient_components(model, layers)

        # encoder module
        self.encoder = create_encoder(self.encoderinputs)

    def get_layer_outputs(self, model, features, inputArray):
        """
        Get the intermediate computations (activations) of 
        the hidden layers of the given target model.
        """
        layers = model.layers
        for l in self.layers_to_exploit:
            x = model.input
            y = layers[l-1].output
            # Model created to get output of specified layer
            new_model = tf.keras.Model(x, y)
            predicted = new_model(features)
            inputArray.append(predicted)

    def get_labels(self, labels):
        """
        Retrieves the one-hot encoding of the given labels.
        """
        ohe_labels = self.attack_utils.one_hot_encoding(
            labels, self.ohencoding)
        return ohe_labels

    def get_loss(self, model, features, labels):
        """
        Computes the loss for given model on given features and labels
        """
        logits = model(features)
        loss = CrossEntropyLoss(logits, labels)

        return loss

    def compute_gradients(self, model, features, labels):
        """
        Computes gradients given the features and labels using the loss
        """
        split_features = self.attack_utils.split(features)
        split_labels = self.attack_utils.split(labels)
        gradient_arr = []
        for (feature, label) in zip(split_features, split_labels):
            with tf.GradientTape() as tape:
                logits = model(feature)
                loss = CrossEntropyLoss(logits, label)
            targetvars = model.variables
            grads = tape.gradient(loss, targetvars)
            # Add gradient wrt crossentropy loss to gradient_arr
            gradient_arr.append(grads)

        return gradient_arr

    def get_gradients(self, model, features, labels, inputArray):
        """
        Retrieves the gradients for each example.
        """
        gradient_arr = self.compute_gradients(model, features, labels)
        batch_gradients = []
        for grads in gradient_arr:
            # gradient_arr is a list of size of number of layers having trainable parameters
            gradients_per_example = []
            for g in self.gradients_to_exploit:
                g = (g-1)*2
                shape = grads[g].shape
                reshaped = (int(shape[0]), int(shape[1]), 1)
                toappend = tf.reshape(grads[g], reshaped)
                gradients_per_example.append(toappend.numpy())
            batch_gradients.append(gradients_per_example)

        # Adding the gradient matrices fo batches
        batch_gradients = np.asarray(batch_gradients)
        splitted = np.hsplit(batch_gradients, batch_gradients.shape[1])
        for s in splitted:
            array = []
            for i in range(len(s)):
                array.append(s[i][0])
            array = np.asarray(array)

            inputArray.append(array)

    def get_gradient_norms(self, model, features, labels):
        """
        Retrieves the gradients for each example
        """
        gradient_arr = self.compute_gradients(model, features, labels)
        batch_gradients = []
        for grads in gradient_arr:
            batch_gradients.append(np.linalg.norm(grads[-1]))
        return batch_gradients

    def compute_input_array(self, model, features, labels):
        """
        Computes and collects necessary inputs for attack model
        """
        # container to extract and collect inputs from target model
        inputArray = []

        # Getting the intermediate layer computations
        if self.layers_to_exploit and len(self.layers_to_exploit):
            self.get_layer_outputs(model, features, inputArray)

        # Getting the one-hot-encoded labels
        if self.exploit_label:
            ohelabels = self.get_labels(labels)
            inputArray.append(ohelabels)

        # Getting the loss value
        if self.exploit_loss:
            loss = self.get_loss(model, features, labels)
            loss = tf.reshape(loss, (len(loss.numpy()), 1))
            inputArray.append(loss)

        # Getting the gradients
        if self.gradients_to_exploit and len(self.gradients_to_exploit):
            self.get_gradients(model, features, labels, inputArray)

        return inputArray

    def compute_input_array_all(self):
        model = self.target_train_model
        logging.info("Preparing attack model input arrays...")
        logging.info("    train members")
        self.m_train_inputs = [self.compute_input_array(
            model, x, y) for (x, y) in self.train_datahandler.mtrain]
        logging.info("    train non-members")
        self.nm_train_inputs = [self.compute_input_array(
            model, x, y) for (x, y) in self.train_datahandler.nmtrain]
        logging.info("    test members")
        self.m_test_inputs = [self.compute_input_array(
            model, x, y) for (x, y) in self.train_datahandler.mtest]
        logging.info("    test non-members")
        self.nm_test_inputs = [self.compute_input_array(
            model, x, y) for (x, y) in self.train_datahandler.nmtest]
        logging.info("    done")

    def forward_pass(self, inputArray):
        """
        Perform a forward pass for the attack model
        """
        attack_outputs = self.attackmodel(inputArray)
        return attack_outputs

    def evaluate_attack_accuracy(self):
        """
        Computes test accuracy of the attack model.
        """
        attack_acc = tf.keras.metrics.Accuracy('attack_acc', dtype=tf.float32)

        for i in range(len(self.m_test_inputs)):
            # Computing the membership probabilities
            mprobs = self.forward_pass(self.m_test_inputs[i])
            nonmprobs = self.forward_pass(self.nm_test_inputs[i])
            probs = tf.concat((mprobs, nonmprobs), 0)

            target_ones = tf.ones(mprobs.shape, dtype=bool)
            target_zeros = tf.zeros(nonmprobs.shape, dtype=bool)
            target = tf.concat((target_ones, target_zeros), 0)

            attack_acc(probs > 0.5, target)

        result = attack_acc.result()
        return result

    def train_attack(self):
        """
        Trains the attack model
        """
        assert self.attackmodel, "Attack model not initialized"
        assert self.m_train_inputs and self.nm_train_inputs and self.m_test_inputs and self.nm_test_inputs, "Input arrays not initialized"

        attack_acc = tf.keras.metrics.Accuracy('attack_acc', dtype=tf.float32)
        attack_accuracy_list = []
        ATTACK_ACCURACY_MOVING_AVG_MAX_SIZE = 20
        best_accuracy = 0.5

        for e in range(self.epochs):
            for i in range(len(self.m_train_inputs)):
                with tf.GradientTape() as tape:
                    tape.reset()
                    # Getting outputs of forward pass of attack model
                    moutputs = self.forward_pass(self.m_train_inputs[i])
                    nmoutputs = self.forward_pass(self.nm_train_inputs[i])
                    # Computing the true values for loss function according
                    memtrue = tf.ones(moutputs.shape)
                    nonmemtrue = tf.zeros(nmoutputs.shape)
                    target = tf.concat((memtrue, nonmemtrue), 0)
                    probs = tf.concat((moutputs, nmoutputs), 0)
                    attackloss = mse(target, probs)
                # Computing gradients
                grads = tape.gradient(
                    attackloss, self.attackmodel.variables)
                self.optimizer.apply_gradients(
                    zip(grads, self.attackmodel.variables))

            if e < self.epochs - ATTACK_ACCURACY_MOVING_AVG_MAX_SIZE:
                # Calculating Attack accuracy
                attack_acc(probs > 0.5, target)
                logging.info(
                    f"Epoch {e + 1} over: train acc {attack_acc.result()}")
            else:
                attack_accuracy = self.evaluate_attack_accuracy()
                attack_accuracy_list.append(attack_accuracy)

                if attack_accuracy > best_accuracy:
                    best_accuracy = attack_accuracy

                attack_accuracy_moving_avg_size = min(
                    len(attack_accuracy_list), ATTACK_ACCURACY_MOVING_AVG_MAX_SIZE)
                attack_accuracy_moving_avg = sum(
                    attack_accuracy_list[-attack_accuracy_moving_avg_size:]) / attack_accuracy_moving_avg_size

                logging.info(
                    f"Epoch {e + 1} over: test acc {attack_accuracy}, "
                    f"moving avg test acc {attack_accuracy_moving_avg}, "
                    f"best test acc {best_accuracy}")

        if type(best_accuracy) is float:
            return attack_accuracy_moving_avg, best_accuracy
        else:
            return float(attack_accuracy_moving_avg.numpy()), float(best_accuracy.numpy())
