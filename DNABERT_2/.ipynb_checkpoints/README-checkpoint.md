# Fine-Tuning DNABERT 
The code was modified to do regression in place of classification 

## Overview

This guide explains how to fine-tune DNABERT on datasets with varying sequence lengths. You can use 4 T4 GPUs for sequences of 4000 base pairs (bp), you can train with 8 sequences per device and evaluate with 16 sequences per device.

## Prerequisites

Before starting the training process, make sure to install the required dependencies:
pip install -r requirements.txt
pip uninstall triton

## Parameters settings

To launch the training, you will need to configure several parameters in the run.bash script:

Dataset Path: Specify the path to the folder containing your datasets with different sequence lengths.

Datasets: Provide the names of the folders that contain the datasets you want to use for experiments.

Max Length: Set the maximum sequence length  each dataset. Ensure that the max_length parameter is set to 25% of the maximum sequence length of your sequences (max_length = sequence_length * 0.25).

Output Path: Specify the path where you want to save the output results.

Batch Size: Define the batch size per device  training and evaluation.

Epochs: Set the number of training epochs.

Other Parameters: Adjust any other parameters as needed, such as learning rate, gradient accumulation steps, etc.

## Finaly 

Once you have configured the run.bash script, you can launch the fine-tuning process by running the following command in the terminal:
bash run.bash

## Loss figures: 
To get the loss figures of fine-tuning, simply copy and paste the notebook that is in finetuning/output_example/runs/Untitled.ipynb into the runs folder in the results directory. Then, you need to replace the folder name in the notebook with the folder name present in runs.