# Federated Learning simulation
Simulation of a Federated Learning scenario using Tensorflow Federated and a federated version of MNIST dataset.
This source code is the result of following several workshops and tutorials about Tensorflow Federated.
The goal is to modularize a bit the code used with Tensorflow Federated, but because of the annotations of the framework, 
that's been a hard task.

## Installation

Just install python 3.8 along _nest_asyncio_ and _Tensorflow Federated_.

    pip install --quiet --upgrade tensorflow_federated
    pip install --quiet --upgrade nest_asyncio
    
## Run the simulation

    cd src
    python simulation.py
    
### Changing configuration

At `src/simulation.py` line 10, change the values of these parameters:

    num_clients=10, batch_size=20, rounds=100 

The most important argument is the _rounds_. More rounds will give more accuracy of the trained model.
You'll be able to see the result printed in the standard output.

    2042/2042 [==============================] - 16s 8ms/step - loss: 1.5411 - sparse_categorical_accuracy: 0.6575
    
    Process finished with exit code 0
