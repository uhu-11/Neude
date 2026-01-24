## Neude

This repository provides the code of the paper "**Bridging Code and Models: A Lightweight Coverage-Guided Fuzzing Framework for Hybrid Deep Learning Systems**"

Neude is a lightweight and extensible coverage-guided fuzzing framework designed specifically for hybrid AI-enabled systems. Neude treats the entire system as a unified testing target, leveraging both traditional code coverage and model-specific coverage to guide input generation. We evaluate Neude on Pylot,  a complex autonomous driving system that tightly couples deep learning models with control logic.

## The structure of the repository

```
Neude
├── pylot/                              Pylot autonomous vehicle platform
│   ├── pylot/
│   ├── configs/                        configuration files for different scenarios
│   ├── dependencies/
│   ├── scripts/
│   ├── data/                           collected data (images, logs)
│   ├── covhtml/                        HTML coverage reports
│   ├── error_seeds/                    error-triggering seeds
│   ├── error_infos/                    error information files
│   ├── local_seeds_pool/               local seed pool for fuzzing
├── pythonfuzz/                         Python fuzzing framework
│   ├── pythonfuzz/                     core fuzzing engine
│   ├── PTtool/                         fuzzing tools
│   ├── ATS/                            automated test suite
│   └── traj-dist/                      trajectory distance calculation
├── scalpel/                            Python static analysis framework
├── tmp/                                temporary files and analysis tools
├── mutilprocess/                       multiprocessing test utilities
└── coverage_test/                      coverage testing directory
```

## Installation

We implement all the Neude systems with Tensorflow 2.5.1 and Python 3.8 in the Linux environment. All experiments are conducted on a server with an Intel i7-10700K CPU (3.80 GHz), 48 GB RAM, and an NVIDIA RTX 3070 GPU (8 GB VRAM).

### Basic Dependency

Run the following command to install the dependencies.

```
cd pylot
pip install -r requirements.txt
```

### Complete Requirements

In order to reproduce our experiment, we should install the complete dependency.

#### Install CALAR Systems

The dataset used for testing is collected using the Pylot data collection script, which gathers data from the Calar system. In addition to the visual data, the runtime information provided by Calar is also recorded, which supports consistent replay and offline analysis. 

The dataset consists of six scenes—Town1, Town2, Town3, Town4, Town5, and Town6—each of which represents a different driving environment. Initial data from each scene is collected as seed data to ensure that the starting conditions are accurate.

1. Download dependencies.

   The dependencies are stored in Baidu Netdisk. Download link:

   ```
   https://pan.baidu.com/s/1KntMPLX-DlbpGOsZGOMzbg 
   Extraction code: y9n4
   ```

2. Set the `CARLA_HOME` environment variable.

   ```
   export CARLA_HOME=./pylot/dependencies/CARLA_0.9.10.1/
   ```

### Quick Start

1. Set the `PYLOT_HOME` environment variable.

```
cd pylot
export PYLOT_HOME=`pwd`/
```

2. Execute the specified script.

```
cd $PYLOT_HOME/scripts/
source ./set_pythonpath.sh
```

3. Run the test.

```
cd  $PYLOT_HOME/
python fuzz_list.py dirs seed_list2.txt --has-model True --use-nc True --batch-size 10 | tee terminal_output.txt
```

Results and key data are saved in JSON format in the newly created `result/datas` directory.

### Running Individual Modules

1. Obstacle Detection Module

```bash
python3 pylot3.py --flagfile=configs/detection2.conf > detection2.txt 2>&1
```

2. Traffic Light Module

```bash
python3 pylot3.py --flagfile=configs/traffic_light.conf  > traffic_light.txt 2>&1
```

3. MPC Control Module

```bash
python3 pylot3.py --flagfile=configs/mpc2.conf > mpc2.txt 2>&1
```

More module configurations can be found and used in `Neude/pylot/configs/`

## Custom Configuration

1. Collect data using a custom dataset

   ```bash
   python3 data_gatherer.py --data_path /media/lzq/D/town5 --simulator_town 2 -log_traffic_lights True --log_obstacles True --log_rgb_camera True --log_trajectories True --log_multiple_object_tracker True  --log_file_name /home/lzq/data_gather.log --tracking_num_steps 10 --log_every_nth_message 10
   ```

2. Adjust fuzzing settings in `Neude/pythonfuzz/pythonfuzz/main.py`

   - Fuzzing settings:

     ```
     --regression, run the fuzzer through set of files for regression or reproduction
     --rss-limit-mb, Memory usage in MB
     --max-input-size, Max input size in bytes
     --close-fd-mask, Indicate output streams to close at startup
     --runs, Number of individual test runs, -1 (the default) to run indefinitely.
     --timeout, If input takes longer then this timeout the process is treated as failure case
     --cov-vec-save-iter, Iterations to save coverage vector
     --has-model, to justify the system is tested using model
     ```

   - Select strategy and save paths, set `config.USE_FUNCTIONS` in `Neude/pythonfuzz/pythonfuzz/config.py`, as well as data save paths such as `config.COV_HTML_PATH`, `config.CRASH_SEED_PATH`, `config.ERROR_INFOS_DIR`, `config.LOCAL_SEED_POOL`, etc.

3. Set the dataset in `Neude/pylot/seed_list2.txt`，including the imgs dataset and the labels for perception, planning, and control modules.

