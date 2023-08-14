import datetime
import logging
from argparse import ArgumentParser, ArgumentTypeError, Namespace
from dataclasses import dataclass
from pathlib import Path

import tensorflow as tf
import tensorflow.keras as keras

from attacks.nasr_2019_whitebox.ml_privacy_meter.attack.meminf import \
    initialize
from attacks.nasr_2019_whitebox.ml_privacy_meter.utils.attack_data import \
    AttackData
from data_utils.datasets import *


def dataset_checker(dataset):

    if dataset not in []:
        raise ArgumentTypeError(f"Dataset not available: {dataset}")

    return dataset


def positive_checker(value):

    value_int = int(value)
    if value_int <= 0:
        raise ArgumentTypeError(f"Value should be positive: {value_int}")

    return value_int


def non_negative_checker(value):

    value_int = int(value)
    if value_int < 0:
        raise ArgumentTypeError(f"Value should be non-negative: {value_int}")

    return value_int


def float_range_checker(value, a, b):

    value_float = float(value)
    if value_float < a or value_float > b:
        raise ArgumentTypeError(
            f"Value should be in [{a}, {b}]: {value_float}")

    return value_float


def parse_args() -> Namespace:
    """Parse arguments."""

    # Create parser
    parser = ArgumentParser(
        prog="evaluate_memb_inf.py",
        description="Evaluate membership inference attack",
    )

    # Add arguments
    parser.add_argument(
        "-lp", "--log_path",
        help=f"Path to log folder.",
        default=""
    )
    parser.add_argument(
        "-d", "--dataset",
        choices=DATASETS,
        help=f"Available datasets: {', '.join(list(DATASETS))}.",
        required=True
    )
    parser.add_argument(
        "-dp", "--data_path",
        help=f"Path to data directory.",
        required=True
    )
    parser.add_argument(
        "-trs", "--train_size",
        type=non_negative_checker,
        help=f"Training dataset size.",
        required=True
    )
    parser.add_argument(
        "-tss", "--test_size",
        type=non_negative_checker,
        help=f"Testing dataset size.",
        required=True
    )
    parser.add_argument(
        "-a", "--architecture",
        nargs="*",
        type=positive_checker,
        help=f"Model architecture (without input and output layers), i.e., list"
              "hidden layers sizes as in -a 256 128 64.",
        required=True
    )
    parser.add_argument(
        "-b", "--batch_size",
        type=non_negative_checker,
        help=f"Batch size for training.",
        default=0
    )
    parser.add_argument(
        "-e", "--epochs",
        type=non_negative_checker,
        help=f"Number of epochs for training.",
        required=True
    )
    parser.add_argument(
        "-lr", "--learning_rate",
        type=float,
        help=f"Learning rate for training.",
        required=True
    )
    # parser.add_argument(
    #     "-el", "--exploit_layers",
    #     nargs="*",
    #     type=int,
    #     help=f"Attack: layers to exploit.",
    #     required=True
    # )
    # parser.add_argument(
    #     "-eg", "--exploit_gradients",
    #     nargs="*",
    #     type=int,
    #     help=f"Attack: gradients to exploit.",
    #     required=True
    # )
    # parser.add_argument(
    #     "-els", "--exploit_loss",
    #     type=bool,
    #     help=f"Attack: exploit loss.",
    #     required=True
    # )
    # parser.add_argument(
    #     "-elb", "--exploit_label",
    #     type=bool,
    #     help=f"Attack: exploit label.",
    #     required=True
    # )
    parser.add_argument(
        "-ae", "--attack_epochs",
        type=non_negative_checker,
        help=f"Number of epochs for the attack.",
        required=True
    )
    parser.add_argument(
        "-ab", "--attack_batch_size",
        type=non_negative_checker,
        help=f"Batch size for the attack.",
        required=True
    )
    parser.add_argument(
        "-ap", "--attack_percentage",
        type=lambda value: float_range_checker(value, 0.0, 1.0),
        help=f"Attack percentage for the attack.",
        required=True
    )
    parser.add_argument(
        "-ar", "--attack_reps",
        type=non_negative_checker,
        help=f"Number of times the attack is repeated (for consistency).",
        required=True
    )

    # Parse arguments and return
    return parser.parse_args()


def generate_model(
    architecture,
    input_shape,
    output_units,
    activation_function=tf.nn.sigmoid,
    max_value=1.0
):

    model = tf.keras.Sequential()

    # Add input layers
    shape = (input_shape,) if type(input_shape) is int else input_shape
    model.add(keras.layers.Input(shape=shape))

    # Add hidden layers
    for layer_size in architecture:
        model.add(
            keras.layers.Dense(
                layer_size,
                activation=activation_function,
                kernel_initializer=keras.initializers.RandomUniform(
                    minval=-max_value, maxval=max_value, seed=None),
                bias_initializer=keras.initializers.RandomUniform(
                    minval=-max_value, maxval=max_value, seed=None)
            )
        )

    # Add output layer
    model.add(
        keras.layers.Dense(
            output_units,
            activation=activation_function,
            kernel_initializer=keras.initializers.RandomUniform(
                minval=-max_value, maxval=max_value, seed=None),
            bias_initializer=keras.initializers.RandomUniform(
                minval=-max_value, maxval=max_value, seed=None)
        )
    )

    return model


def save_model(model, filename):
    weights = model.get_weights()
    with open(filename, "w") as f:
        counter = 0
        for w, b in zip(weights[0::2], weights[1::2]):
            f.write(f"w{counter}={w.tolist()}\n")
            f.write(f"b{counter}={b.tolist()}\n")
            counter += 1


class LoggerCallback(keras.callbacks.Callback):
    def __init__(self):
        super(LoggerCallback, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def on_epoch_end(self, epoch, logs=None):
        metrics = [f"{key}: {value:.4f}" for key, value in logs.items()]
        log_message = f"Epoch {epoch + 1}: {' - '.join(metrics)}"
        self.logger.info(log_message)


@dataclass
class AttackConfig:
    layers_to_exploit: List[int]
    gradients_to_exploit: List[int]
    exploit_loss: bool
    exploit_label: bool


def main() -> None:
    """Main method executed when program is run."""

    ########################################################################
    #              Parsing arguments and initializing logger               #
    ########################################################################

    # Parse arguments
    args = parse_args()

    # Configure logger
    current_time = datetime.datetime.now()
    timestamp = current_time.strftime("%Y%m%d_%H%M%S_%f")
    log_filename = f"eval_memb_inf_{timestamp}.log"
    log_filename = Path(args.log_path).joinpath(log_filename)
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format="%(message)s"
    )

    # Log arguments
    for arg in vars(args):
        logging.info(f"{arg}={getattr(args, arg)}")

    ########################################################################
    #                        Training target model                         #
    ########################################################################

    # Load dataset
    dataset = DATASETS[args.dataset]
    logging.info(dataset)
    train_x, train_y, test_x, test_y = dataset.load(
        args.data_path,
        train_size=args.train_size,
        test_size=args.test_size,
        to_categorical=False
    )
    train_y_ohe = keras.utils.to_categorical(train_y, dataset.num_classes)
    test_y_ohe = keras.utils.to_categorical(test_y, dataset.num_classes)

    # Initialize model
    model = generate_model(
        args.architecture,
        dataset.feature_shape,
        dataset.num_classes
    )

    # Compile model
    model.compile(
        optimizer=keras.optimizers.SGD(args.learning_rate),
        loss=keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.CategoricalAccuracy()]
    )

    # Train model
    logger_callback = LoggerCallback()
    model.fit(
        train_x, train_y_ohe,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_data=(test_x, test_y_ohe),
        callbacks=[logger_callback],
        verbose=0
    )

    # Save model
    model_filename = f"eval_memb_inf_{timestamp}.model"
    model_filename = Path(args.data_path).joinpath("models", model_filename)
    save_model(model, model_filename)

    ########################################################################
    #                        Attacking target model                        #
    ########################################################################

    model_depth = len(args.architecture) + 1
    configurations: List[AttackConfig] = []
    for i in range(1, model_depth):
        configurations.append(AttackConfig([i], [], False, False))
        configurations.append(AttackConfig([i], [i], False, True))
    configurations.append(AttackConfig([model_depth], [], False, False))
    configurations.append(AttackConfig(
        [model_depth], [model_depth], True, True))

    # Create attack datahandler
    train_datahandler = AttackData(
        train_x, train_y,
        test_x, test_y,
        num_class=None,
        attack_percentage=args.attack_percentage,
        batch_size=args.attack_batch_size
    )

    for config in configurations:

        logging.info(config)

        attackobj = initialize(
            target_train_model=model,
            target_attack_model=model,
            train_datahandler=train_datahandler,
            attack_datahandler=None,
            layers_to_exploit=config.layers_to_exploit,
            gradients_to_exploit=config.gradients_to_exploit,
            exploit_loss=config.exploit_loss,
            exploit_label=config.exploit_label,
            epochs=args.attack_epochs
        )

        attackobj.compute_input_array_all()

        best_accuracy_overall = 0.0
        avg_accuracy_list = []

        # Attack
        for attack_counter in range(1, args.attack_reps + 1):

            logging.info("Starting attack attempt {}".format(attack_counter))

            attackobj.initialize_attack_model()

            avg_accuracy, best_accuracy = attackobj.train_attack()

            best_accuracy_overall = max(best_accuracy_overall, best_accuracy)
            avg_accuracy_list.append(avg_accuracy)

            logging.info(f"Attack attempt {attack_counter}/{args.attack_reps}\
                finished: avg accuarcy {avg_accuracy}, best accuracy {best_accuracy}\n")

        avg_accuracy_overall = sum(avg_accuracy_list) / len(avg_accuracy_list)
        logging.info(
            f"Attack finished: avg accuracy overall {avg_accuracy_overall}, "
            f"best accuracy overall {best_accuracy_overall}\n"
        )


if __name__ == "__main__":
    main()

# python .\src\evaluate_memb_inf.py -lp logs -d mnist -dp data -trs 100 -tss 100 -a 30 20 -b 10 -e 200 -lr 3.0 -ae 100 -ab 10 -ap 0.50 -ar 4
# python .\src\evaluate_memb_inf.py -lp logs -d purchase100 -dp data -trs 1000 -tss 1000 -a 1024 512 256 128 -b 2 -e 200 -lr 30.0 -ae 100 -ab 10 -ap 0.50 -ar 4
# python .\src\evaluate_memb_inf.py -lp logs -d texas100 -dp data -trs 1000 -tss 1000 -a 1024 512 256 128 -b 2 -e 200 -lr 30.0 -ae 100 -ab 10 -ap 0.50 -ar 4
# python .\src\evaluate_memb_inf.py -lp logs -d location -dp data -trs 1200 -tss 2000 -a 256 128 64 -b 5 -e 100 -lr 30.0 -ae 100 -ab 10 -ap 0.50 -ar 4
