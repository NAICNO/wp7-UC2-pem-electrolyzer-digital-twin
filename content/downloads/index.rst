Downloads & Quick Reference
===========================

.. important::
   This repository is self-contained. After cloning, run ``./setup.sh`` to set up the environment automatically.

Setup Script
------------

Run after cloning to set up the environment:

.. code-block:: bash

   git clone https://github.com/NAICNO/wp7-UC2-pem-electrolyzer-digital-twin.git
   cd wp7-UC2-pem-electrolyzer-digital-twin
   ./setup.sh

Available Models
----------------

.. list-table::
   :header-rows: 1
   :widths: 20 30 15 35

   * - Name
     - Class
     - Parameters
     - Description
   * - ``teacher``
     - ``HybridPhysicsMLP``
     - 9,354
     - 8 physics params + MLP residual correction
   * - ``student``
     - ``PhysicsHybrid12Param``
     - 12
     - 6 physics + 6 hybrid correction params
   * - ``pure_mlp``
     - ``PureMLP``
     - 8,961
     - No-physics MLP baseline
   * - ``big_mlp``
     - ``BigMLP``
     - 43,393
     - Large no-physics MLP baseline
   * - ``transformer``
     - ``SteadyStateTransformer``
     - 529,793
     - Self-attention baseline

Command Line Options
--------------------

.. code-block:: bash

   python scripts/pem_electrolyzer/main.py [OPTIONS]

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Option
     - Default
     - Description
   * - ``--mode``
     - ``full``
     - Execution mode: ``full``, ``quick-test``, ``teacher-only``, ``ablation``
   * - ``--epochs``
     - ``100``
     - Number of training epochs
   * - ``--batch-size``
     - ``4096``
     - Training batch size
   * - ``--lr``
     - ``0.01``
     - Learning rate
   * - ``--device``
     - ``auto``
     - Device selection: ``cuda``, ``cpu``, or ``auto``
   * - ``--seed``
     - ``42``
     - Random seed for reproducibility
   * - ``--data-dir``
     - ``dataset/``
     - Path to dataset directory
   * - ``--results-dir``
     - ``results/``
     - Path to output directory

Available Datasets
------------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Dataset
     - Description
   * - ``dataset/test4_subset.csv``
     - Training data -- long-term stability test
   * - ``dataset/test2_subset.csv``
     - OOD evaluation -- current sweep
   * - ``dataset/test3_subset.csv``
     - OOD evaluation -- pressure swap

Example Commands
----------------

**Quick Test (< 1 minute)**

.. code-block:: bash

   python scripts/pem_electrolyzer/main.py --mode quick-test

**Full Training (GPU)**

.. code-block:: bash

   python scripts/pem_electrolyzer/main.py --mode full --device cuda --epochs 100

**Teacher Only**

.. code-block:: bash

   python scripts/pem_electrolyzer/main.py --mode teacher-only --epochs 50

**Background Training (tmux)**

.. code-block:: bash

   tmux new -s training 'python scripts/pem_electrolyzer/main.py \
       --mode full --epochs 100 2>&1 | tee training.log'

   # Monitor: tail -f training.log
   # Attach: tmux attach -t training

For AI Coding Assistants
------------------------

If you're using an AI coding assistant (Claude Code, GitHub Copilot, Cursor, etc.), the repository includes machine-readable instruction files:

- ``AGENT.md`` -- Markdown format (human and agent readable)
- ``AGENT.yaml`` -- YAML format (structured data for programmatic parsing)

These files contain step-by-step instructions that agents can follow to:

1. Set up the environment on the VM
2. Run the Jupyter notebook
3. Execute command-line experiments
4. Verify results

**Quick prompt for your AI assistant:**

.. code-block:: text

   Read AGENT.md and help me run the PEM electrolyzer PINN
   demonstrator on my NAIC VM.
   VM IP: <your_vm_ip>
   SSH Key: <path_to_your_key.pem>

The agent will execute the setup and run experiments based on the structured instructions.
