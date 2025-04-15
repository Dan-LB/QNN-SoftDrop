# **Softer is Better: Tweaking Quantum Dropout to Enhance Quantum Neural Network Trainability**

Accepted at the International Conference on Quantum Communications, Networking, and Computing ([QCNC 2025](https://www.ieee-qcnc.org/2025/))

**Authors:**  
- Daniele Lizzio Bosco           :abacus: :dna: :envelope: 
- Riccardo Romanello   :chart_with_upwards_trend: :envelope: 
- Giuseppe Serra   :abacus:
- Carla Piazza :abacus:




:abacus: Department of Mathematics, Computer Science and Physics, University of Udine, Udine, Italy 

:dna:  Department of Biology, University of Naples Federico II, Napoli, Italy

:chart_with_upwards_trend: Department of Environmental Sciences, Informatics and Statistics, Ca’ Foscari University of Venice, Venice, Italy

:envelope: Corresponding authors [lizziobosco.daniele@spes.uniud.it], [riccardo.romanello@unive.it]

**Read the paper here:**  
WORK IN PROGRESS

In this paper, we propose a novel technique to reduce overfitting in Quantum Neural Networks (QNNs) by generalizing quantum dropout. In particular, we introduce a "soft version" of dropout that leverage the continuity of quantum operators to "smooth" the remotion of a gate during training. We apply our method for a variety of dropout strategies on a simple function approximation task, showing in general a consistent advantage for all dropout strategies involving entangling gates. 


## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
2. [Experiments and Results](#experiments-and-results)
3. [Cite this work](#cite-this-work)

## Introduction

This project explores a novel approach to enhance the training of quantum neural networks by using a modified dropout method we called "soft dropout." By selectively dropping out units during the training process, the model aims to increase robustness and improve generalization, which are crucial in quantum machine learning applications. 

This README will guide you through setting up the environment and running experiments.


## Installation

To set up the environment and install all necessary dependencies, follow these steps:

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  
   ```

3. Install dependencies from `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```
   This will install the following packages:
         <details>
    <summary>Requirements list</summary>
    
    ```yaml
   appnope==0.1.4
   asttokens==2.4.1
   attrs==23.2.0
   backcall==0.2.0
   beautifulsoup4==4.12.3
   bleach==6.1.0
   certifi==2024.2.2
   charset-normalizer==3.3.2
   contourpy==1.2.0
   cycler==0.12.1
   decorator==5.1.1
   defusedxml==0.7.1
   dill==0.3.8
   docopt==0.6.2
   executing==2.0.1
   fastdtw==0.3.4
   fastjsonschema==2.19.1
   fonttools==4.49.0
   idna==3.6
   ipython==8.12.3
   jedi==0.19.1
   Jinja2==3.1.3
   joblib==1.3.2
   jsonschema==4.21.1
   jsonschema-specifications==2023.12.1
   jupyter_client==8.6.0
   jupyter_core==5.7.1
   jupyterlab_pygments==0.3.0
   kiwisolver==1.4.5
   MarkupSafe==2.1.5
   matplotlib==3.8.3
   matplotlib-inline==0.1.6
   mistune==3.0.2
   mpmath==1.3.0
   nbclient==0.9.0
   nbconvert==7.16.1
   nbformat==5.9.2
   numpy==1.26.4
   packaging==23.2
   pandocfilters==1.5.1
   parso==0.8.3
   pbr==6.0.0
   pexpect==4.9.0
   pickleshare==0.7.5
   pillow==10.2.0
   pipreqs==0.5.0
   platformdirs==4.2.0
   prompt-toolkit==3.0.43
   psutil==5.9.8
   ptyprocess==0.7.0
   pure-eval==0.2.2
   Pygments==2.17.2
   pyparsing==3.1.1
   python-dateutil==2.8.2
   pyzmq==25.1.2
   qiskit==1.1.0
   qiskit-algorithms==0.3.0
   qiskit-machine-learning==0.7.1
   referencing==0.33.0
   requests==2.31.0
   rpds-py==0.18.0
   rustworkx==0.14.0
   scikit-learn==1.4.1.post1
   scipy==1.12.0
   six==1.16.0
   soupsieve==2.5
   stack-data==0.6.3
   stevedore==5.1.0
   symengine==0.11.0
   sympy==1.12
   threadpoolctl==3.3.0
   tinycss2==1.2.1
   tornado==6.4
   traitlets==5.14.1
   typing_extensions==4.9.0
   urllib3==2.2.1
   wcwidth==0.2.13
   webencodings==0.5.1
   yarg==0.1.9
   torch==2.5.0
   latex==0.7.0

    ```

    </details>


## Experiments and Results

The project includes a series of experiments to evaluate the effectiveness of the soft dropout method in enhancing QNN performance. Each experiment tests different configurations of the dropout rate, network architecture, and training settings.

To reproduce the main experiment:

1. Follow the setup and installation instructions.
2. Run the provided scripts in the `regression_single.py` file. You can modify the parameter "softness" and "probability" according to number used in the paper to test different configurations. It is also possible to modify the dropout strategy.

After running the experiments, results will be stored in the `results` folder for easy access and analysis.

The plots shown in Figure 4 and Figure 5 can be obtained by the corresponding notebooks.

## Cite this work
Work in progress