# **Project Title: Sofer is Better: Tweaking Quantum Dropout to Enhance Quantum Neural Network Trainability**

This project implements soft dropout for quantum neural networks (QNN). Soft dropout is a technique used to improve the generalization of neural networks by randomly dropping units during training.

## **Introduction**

This project explores a novel approach to enhance the training of quantum neural networks by using a modified dropout method we called "soft dropout." By selectively dropping out units during the training process, the model aims to increase robustness and improve generalization, which are crucial in quantum machine learning applications. 

This README will guide you through setting up the environment and running experiments.

A more complete README with all the experiments is under construction, and will be published after the review process.


## **Installation**

To set up the environment and install all necessary dependencies, follow these steps:

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  
   ```

3. Install dependencies from `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```


## **Experiments and Results**

The project includes a series of experiments to evaluate the effectiveness of the soft dropout method in enhancing QNN performance. Each experiment tests different configurations of the dropout rate, network architecture, and training settings.

To reproduce the experiments:

1. Follow the setup and installation instructions.
2. Run the provided scripts in the `regression_single.py` file or any other relevant experiment scripts to generate results. You can modify the parameter "softness" and "probability" according to number used in the paper to test different configurations.

After running the experiments, results will be stored in the `results` folder for easy access and analysis. Key metrics and visualizations of the results will be available here to help interpret the effects of soft dropout on QNN trainability.

