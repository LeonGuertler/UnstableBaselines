Installation
============

You can install Unstable Baselines via PyPi:

.. code-block:: bash

    pip install unstable-baselines

or build the project from source:

.. code-block:: bash

    git clone https://github.com/unstable-baselines/unstable-baselines.git
    cd unstable-baselines
    pip install -e .

Our `main`-branch uses DeepSpeed and FlashAttention for a faster training experience.
If you want a plain implementation for even lighter dependencies and prototyping, visit our `legacy`-branch.

Multi-GPU Setup
"""""""""""""""

GPU-use is required for asynchronous data collection while game-playing and model training. 
You can define how many GPUs to use for the learner which updates the parameters with config parameter:

.. code-block:: yaml
   :linenos:

   leaner: 
        num_gpus: 2

The rest of the available ressources will be allocated by the game scheduler and its workers to collect game data.
To restrict GPU-use, set the environment variable `CUDA_VISIBLE_DEVICES` before running your script:   

.. code-block:: bash

    export CUDA_VISIBLE_DEVICES=0,1
    python examples/run.py

Multi-Node Setup
""""""""""""""""

While it is technically possible to distribute the components of Unstable Baselines across multiple nodes,
this project is primarily designed for single-node setups and rapid prototyping.
For highly optimized multi-node deployments, we recommend exploring other established frameworks, such as VERL and Verifiers,
and building custom solutions tailored to your specific requirements.

If you still wish to run Unstable Baselines in a multi-node setup, start by setting up the master node using:

.. code-block:: bash

    ray start --head

Then, on each worker node, connect to the master node with:

.. code-block:: bash

    ray start --address='[MASTER_NODE_IP]:6379'

Replace `[MASTER_NODE_IP]` with the actual IP address of your master node.
Finally, run your Unstable Baselines script on the master node as usual.
