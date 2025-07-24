# nlp-anomalous-gibberish

## Project Overview

This repository contains an nlp-anomalous-gibberish text dettection project with three different models and input data files. The project is organized to keep data, models, and outputs separated for clarity and ease of use.

## Repository Structure

/
├── data/
│ ├── data_expanded.xlsx # Excel input data file
│ └── train.txt # Text input data file for training the Markov model
│
├── GPT2/
│ ├── gpt2_main.py # Main script for gpt2 model
│ └── outputs/ # Output folder for gpt2 model (Excel, PNG)
│     ├── gpt2_results.xlsx
│     └── #png files
│
├── Markov chain/
│ ├── markov_main.py # Main script for Markov model
│ ├── plot.py # Plotting script for Markov model
│ └── outputs/ # Output folder for Model 2 results
│     ├── markov_results.xlsx
│     └── #png files
│
├── Tiny Llama/
│ ├── tinyllama_main.py # Main script for Tiny Llama model
│ ├── plot.py # Plotting script for Tiny Llama model
│ └── outputs/ # Output folder for Tiny Llama model
│     ├── tinyllama_results.xlsx
│     └── #png files
│
├── requirements.txt         # Python dependencies│
└── README.md # This file


## Usage

1. Input files under the `data` folder.
2. Run the main script inside the model folder  (e.g., `GPT2/gpt2_main.py`).
3. If available, run the plotting script to generate visualizations.
4. Outputs such as result Excel files and plots will be saved inside the respective model's `outputs` folder.

## Requirements : requirements.txt

