import numpy as np
import tensorflow as tf


class FederatedSimulator:
    def __init__(self, clients_training_data, clients_test_data, batch_format_fn, create_keras_model_fn, num_clients=10, batch_size=20, rounds=50):
        np.random.seed(0)
        # TODO arguments of the simulator
        self.num_clients = num_clients
        self.batch_size = batch_size
        self.rounds = rounds
        self.__batch_format_fn = batch_format_fn
        self.__create_keras_model = create_keras_model_fn
        self.__federated_training_data = self.__build_federated_training_data(clients_training_data)
        self.__federated_test_data = self.__preprocess(clients_test_data.create_tf_dataset_from_all_clients())
        self.__server_state = None

    def run_simulation(self, federated_algorithm):
        self.__server_state = federated_algorithm.initialize()
        for round in range(self.rounds):
            self.__server_state = federated_algorithm.next(self.__server_state, self.__federated_training_data)

    def evaluate(self):
        keras_model = self.__create_keras_model()
        keras_model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
        )
        keras_model.set_weights(self.__server_state)
        keras_model.evaluate(self.__federated_test_data)

    def get_federated_training_data(self):
        return self.__federated_training_data

    def __preprocess(self, dataset):
        return dataset.batch(self.batch_size).map(self.__batch_format_fn)

    def __build_federated_training_data(self, training_data):
        client_ids = np.random.choice(training_data.client_ids, size=self.num_clients, replace=False)
        federated_training_data = [self.__preprocess(training_data.create_tf_dataset_for_client(x))
                                   for x in client_ids
                                   ]
        return federated_training_data
