# DExTER

Christophe Menichelli, Vincent Guitard, Sophie Lèbre, Jose-Juan Lopez-Rubio, Charles-Henri Lecellier, Laurent Bréhélin (2020)


## Installation

 * DExTER code source: ```git clone https://gite.lirmm.fr/menichelli/DExTER.git```

 * python 3.6+ : we recommend to use anaconda3 (www.anaconda.com)
   * Required packages: numpy (1.18.1), tqdm (4.42.0), matplotlib (3.1.2), pandas (1.0.0), scikit-learn (0.22.1), bitarray (1.2.1)
   * Depending on your distribution, most of these packages should already be available on anaconda3. To install the missing packages: ```~/anaconda3/bin/pip3 install [package]```

 * graphviz : optional. If installed, this program will be used to generate exploration graphs and lattices images. On Ubuntu it can be installed by: ```sudo apt install graphviz```


## Parameters

```
usage: Main.py [-h] -fasta_file FASTA_FILE
               [-alignement_point_index ALIGNEMENT_POINT_INDEX]
               -expression_file EXPRESSION_FILE -target_condition
               TARGET_CONDITION [-log_transform]
               [-experience_directory EXPERIENCE_DIRECTORY] [-cache]
               [-verbose] [-kmer_max_length KMER_MAX_LENGTH] -nb_bins NB_BINS
               [-uniform_bins] [-correlation_increase CORRELATION_INCREASE]
               [-correlation_increase_ratio CORRELATION_INCREASE_RATIO]
               [-correlation_min CORRELATION_MIN] [-final_lasso]
               [-no_progress] [-random_seed RANDOM_SEED]
               [-nb_thread NB_THREAD] [-time]

DExTER

arguments:
  -h, --help            show this help message and exit
  -fasta_file FASTA_FILE
                        Fasta file of studied sequences. All sequences must
                        have the same length.
  -alignement_point_index ALIGNEMENT_POINT_INDEX
                        Position of the alignment point in sequences (default:
                        0)
  -expression_file EXPRESSION_FILE
                        Expression file associated with sequences. The first
                        column provides the name of the sequences. The other
                        columns give the expression values associated with
                        different conditions. The first line provides the name
                        of the different conditions.
  -target_condition TARGET_CONDITION
                        Name of the target condition to run the analysis
                        (column name in expression file)
  -log_transform        Log transform expression data
  -experience_directory EXPERIENCE_DIRECTORY
                        Directory for storing results (default: ./experience)
  -cache                Store k-mer occurrences
  -verbose              Print img for each k-mer
  -kmer_max_length KMER_MAX_LENGTH
                        Set maximal k-mer length allowed (0 for None, default)
  -nb_bins NB_BINS      Number of bins for segmenting sequences
  -uniform_bins         Bins have all the same length (by default, a
                        polynomial generator is used such that bins close to
                        the alignment point are smaller than bins far from the
                        align point; cf. publication)
  -correlation_increase CORRELATION_INCREASE
                        Minimum correlation difference for continuing
                        exploration (rho_new - rho_old, default: 0.01 [i.e.
                        1%])
  -correlation_increase_ratio CORRELATION_INCREASE_RATIO
                        Minimum correlation difference as ratio of previous
                        correlation ((rho_new-rho_old)/rho_old, default: 0.10
                        [ie: 10%])
  -correlation_min CORRELATION_MIN
                        Stop exploration when correlation passes below this
                        value (default: 0)
  -final_lasso          Fit a linear model with the variables identified
                        during exploration
  -no_progress          Hide progress bar
  -random_seed RANDOM_SEED
                        Seed used for random generator and selection of the
                        learning/test sets (default: 686185200). The same seed
                        must be used when analysing the different conditions
                        of a series to ensure that the same learning/test sets 
						are used in all conditions.
						
  -nb_thread NB_THREAD  Maximal number of thread (-1 for max, default)
  -time                 Print time spent in each step
```


Arguments inside brackets [ ] are optional.


## Outputs

The variables identified are stored in a directory named from the specified condition. Each directory is organized as follows:

	* data/
        - expression data of the condition (text and binary version of these data)
    * img/
        - segments.lattice: lattice representing the regions considered in the exploration
        - this directory also contains the lattices of the tested k-mer sorted in sub-directories named according to k-mer length
    * training_set.log and testing_set.log
        - lists of the genes used for training and testing. Variable extraction and model fitting (if the argument final_lasso has been specified) are done using the training set only.
    * models/
        - contains the values of the identified variables for each gene. One matrix contains the values for the training set while the other one contains the values for the test set.
    * ALL_domains.dat
        - lists of all identified variables.
    * ALL.lasso
        - output and accuracy of the trained linear model (if argument final_lasso has been specified)
    * exploration_graph.svg
        - image of the exploration graph (this file is generated only if program  graphviz is installed)
    * exploration_graph.dat
        - script used to generate exploration_graph.svg


## Example

See file run_example.bash in directory example/

## R training and visualization

Once DExTER has been run on each condition, and predictive variables (i.e. LREs) have been identified, an Rmarkdown script can be used to fit a linear model on each condition and to reproduce the main experiments described in the paper. This script:

  1. provides some statistics about the gene expression data given in input
  2. runs glmnet in multitask learning (see below) to learn a linear model for each condition using the genes in the training set
  3. computes the accuracy of the different models on the test set
  4. runs the permutation experiments to assess the accuracy of the different models on the different conditions (see paper)
  5. plots the importance and correlations with expression of the 5 most important variables of each model (see paper for details)

Models are fit by multitask learning. This means that all models are learned simultaneously with a global penalization. In this way, the same variables are used in every models, but the weight associated with each variable differs depending on the conditions.

 * Required libraries: rmarkdown, tinytex, glmnet, gplots, ggplot2, ggrepel, reshape2, zoo, RColorBrewer (```in R session, type install.packages("[package]")```)

 * To run the script
  1. copy file DExTER_experiments.Rmd into the directory where results are stored (specified by the argument "experience_directory"):
  ```
  ./P_falciparum/
  ├── 0h
  ├── 16h
  ├── 24h
  ├── 32h
  ├── 40h
  ├── 48h
  ├── 8h
  └── DExTER_experiments.Rmd
  ```
  2. open DExTER_experiments.Rmd with Rstudio, set working directory
  3. Knit > knit to pdf

For further information about Rmarkdown in Rstudio, see https://rmarkdown.rstudio.com/articles_intro.html


## Contact

    Christophe Menichelli: christophe.menichelli@lirmm.fr
    Laurent Bréhélin: laurent.brehelin@lirmm.fr


## Known bugs

    None declared.
    
    "Sometimes the problem is to discover what the problem is."  - Gordon Glegg
