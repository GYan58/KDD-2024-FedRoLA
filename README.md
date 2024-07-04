# KDD-2024-FedRoLA

This repository provides the design and implementation details for our proposed method, FedRoLA.

# Usage

Prerequisites
- Python 3.5+
- PyTorch
- CUDA environment

# Directory Structure

1. ./Main.py: Contains configuration settings and the basic framework for Federated Learning.
2. ./Sim.py: Describes simulators for clients and the central server.
3. ./Utils.py: Includes necessary functions and provides guidance on obtaining training and testing data.
4. ./Settings.py: Specifies the required packages and settings.
5. ./Attacks.py: Contains the code for model poisoning attack algorithms.
6. ./Defenses.py: Inlcudes the code for defense algorithms.

# Implementation

1. To execute the algorithms, run the ./Main.py file using the following command:
```
   python3 ./Main.py
```

2. Adjust the parameters and configurations within the ./Main.py file to suit your specific needs.

# Citation

If you use the simulator or some results in our paper for a published project, please cite our work by using the following bibtex entry

```
@inproceedings{yan2024fedrola,
  title={FedRoLa: Robust Federated Learning Against Model Poisoning via Layer-based Aggregation},
  author={Gang Yan, Hao Wang, Xu Yuan and Jian Li},
  booktitle={Proc. of ACM SIGKDD},
  year={2024}
}
```
