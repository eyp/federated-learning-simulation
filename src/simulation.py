import tensorflow as tf
import tensorflow_federated as tff
import nest_asyncio
from federated_simulator import FederatedSimulator
from federated_external_model import batch_format, create_keras_model

nest_asyncio.apply()
emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()

federated_simulator = FederatedSimulator(emnist_train, emnist_test, batch_format, create_keras_model,
                                         num_clients=10, batch_size=20, rounds=100)
federated_training_data = federated_simulator.get_federated_training_data()


@tf.function
def client_update(model, dataset, server_weights, client_optimizer):
    """The most important function, It's the training of each client."""
    # Initialize client weights with server weights.
    client_weights = model.weights.trainable
    tf.nest.map_structure(lambda x, y: x.assign(y),
                          client_weights, server_weights)

    # For each batch in the dataset, compute the gradients using the client optimizer
    for batch in dataset:
        with tf.GradientTape() as tape:
            outputs = model.forward_pass(batch)

        grads = tape.gradient(outputs.loss, client_weights)
        grads_and_weights = zip(grads, client_weights)
        client_optimizer.apply_gradients(grads_and_weights)

    return client_weights


@tf.function
def server_update(model, mean_client_weights):
    """Updates the server weights with an average of the client wegiths calculated by each client"""
    # Get the model weights
    model_weights = model.weights.trainable
    # Assign the mean of the clients weights to the server model weights
    tf.nest.map_structure(lambda x, y: x.assign(y),
                          model_weights, mean_client_weights)
    return model_weights


def model_fn():
    """Creates the Keras model with a loss function, accuray as metric and the specification of the input data"""
    keras_model = create_keras_model()
    return tff.learning.from_keras_model(
        keras_model,
        input_spec=federated_training_data[0].element_spec,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )


@tff.tf_computation
def server_init():
    """Initialization of the server model"""
    model = model_fn()
    return model.weights.trainable


dummy_model = model_fn()

# Definition of Federated Types for Federated Functions
# The arguments of some of the annotations are special types that define the type of the data and where it's used (server or client).
tf_dataset_type = tff.SequenceType(dummy_model.input_spec)
model_weights_type = server_init.type_signature.result
federated_server_type = tff.FederatedType(model_weights_type, tff.SERVER)
federated_dataset_type = tff.FederatedType(tf_dataset_type, tff.CLIENTS)


# Now come the federated functions annotated with Tensorflow Federated special annotations.
# These functions are used by the framework to run the simulation.
# Each federated function uses the corresponding regular function defined previously.
@tff.federated_computation
def initialize_fn():
    return tff.federated_value(server_init(), tff.SERVER)


@tff.tf_computation(tf_dataset_type, model_weights_type)
def client_update_fn(tf_dataset, server_weights):
    model = model_fn()
    client_optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
    return client_update(model, tf_dataset, server_weights, client_optimizer)


@tff.tf_computation(model_weights_type)
def server_update_fn(mean_client_weights):
    model = model_fn()
    return server_update(model, mean_client_weights)


@tff.federated_computation(federated_server_type, federated_dataset_type)
def next_fn(server_weights, federated_dataset):
    # Broadcast the server weights to the clients.
    server_weights_at_client = tff.federated_broadcast(server_weights)

    # Each client computes their updated weights.
    client_weights = tff.federated_map(
        client_update_fn, (federated_dataset, server_weights_at_client)
    )

    # The server averages these updates.
    mean_client_weights = tff.federated_mean(client_weights)

    # The server updates its model.
    server_weights = tff.federated_map(server_update_fn, mean_client_weights)

    return server_weights

# The Federated Lerning algorithm is an 'Iterative Process' which first initializes the server,
# then run next_fn the number of rounds defined at the beginning of the simulation.
federated_algorithm = tff.templates.IterativeProcess(
    initialize_fn=initialize_fn,
    next_fn=next_fn
)
federated_simulator.run_simulation(federated_algorithm)
federated_simulator.evaluate()
