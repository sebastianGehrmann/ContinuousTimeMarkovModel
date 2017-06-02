# Unsupervised Disease Progression

This model contains a reimplementation of the unsupervised disease progression model by Wang et al. [link](https://pdfs.semanticscholar.org/8061/f7d20951a5117855aa08427087a921429d53.pdf). 
The code is heavily based on a previous reimplementation found [here](https://github.com/evidation-health/ContinuousTimeMarkovModel). 

## Installation

The code requires pymc3 (>3.0), please install it using `pip install pymc3==3.0`. We include testdata from the original implementation. 
To check whether the code works for you, please execute `python main.py -t 3 -n 5`. 
This will run the model for only very few data points and iterations. 

## How to use the code

This code requires a CSV file with three columns with the names *PID* (patient ID), *time_delta*, and *Code* sorted by PID and time_delta. 
The patient ID should be an increasing unique integer for each patient. 
time_delta describes the time interval in which an observation was made. The original paper describes a window of 90 days.
Code is a task-dependend observation. While the original paper (and the test data) uses ICD9 as observations, the code will run as long as observations are represented as unique strings.

### Preprocess

Given the CSV file described above, you can preprocess by using `preprocess.py`. The execution requires a number of parameters described in the following list:

- The path to the CSV file (not optional)
- **Outdir** Name of the directory to store preprocessed files in
- **Paramdir** You can use pre-initialized parameters, stored in `.txt` files in this folder. Look at `data/param_init_small` for examples
- **anchorsdir**  The original paper enables the use of anchors for comorbidities. Those are specified in a file named `anchor_icd9.csv` in this directory. 
- **maxclaims** If set, the preprocessing drops all codes that are not in the top n occurring ones
- **minsteps** The minimum number of time windows a patients needs to have data for
- **seed** Random seed
- **kcomorbid** Number of comorbidities
- **mstates** Number of states 

Running the preprocessing will create a number of `.pkl` files that are used in further steps. 

### Running a model

The code to run the model is both in `Model.ipynb` and in `main.py`. We recommend the ipython notebook to become familiar with the code. 
Individual steps of the model are commented. 


### Evaluating a model

The resulting parameters of the model are found in the trace, namely in the last and final step of the trace. 
`Model.ipynb` and `generate chf patients.ipynb` have the necessary code to generate the figures from the original paper. 



