import argparse
import logging
import os

properties_wrapper_instance = None


class PropertiesWrapper:

    def __init__(self, args):
        self.motif_min_size = None
        self.sequence_size = None
        self.index_file = None
        self.expression_file = None
        self.target_condition = None
        self.log_transform = None
        self.fasta_file = None
        self.experience_directory = None
        self.alignement_point_index = None
        self.nb_thread = None
        self.use_iupac = None
        self.use_gap = None
        self.cache = None
        self.verbose = None
        self.no_progress = None
        self.nb_regions = None
        self.random_seed = None
        self.correlation_increase = None
        self.correlation_increase_ratio = None
        self.correlation_min = None
        self.kmer_max_length = None
        self.init_lasso = None
        self.final_lasso = None
        self.logistic = None
        self.regions_fixed_length = None
        self.time = None
        self.set_bins = None
        self.build_properties(args)
        self.graph_printer = None
        self.dct_gene_expr = None
        self.dct_lasso_residual_error = None
        self.training_set = None
        self.testing_set = None
        self.fasta_parser = None

    def build_properties(self, args):
        parser = argparse.ArgumentParser(description='DExTER')
        parser.add_argument('-fasta_file', type=str, help='Fasta file of studied sequences. All sequences must have the same length.', required=True)
        parser.add_argument('-alignement_point_index', type=int, help='Position of the alignment point in sequences (default: 0)', default=0)
        # parser.add_argument('-index_file', type=str, help='path to the index (.ssa) file', required=True)  # not used anymore
        parser.add_argument('-index_file', type=str, help=argparse.SUPPRESS)
        parser.add_argument('-expression_file', type=str, help='Expression file associated with sequences. The first column provides the name of the sequences. The other columns give the expression values associated with different conditions. The first line provides the name of the different conditions.', required=True)
        parser.add_argument('-target_condition', type=str, help='Name of the target condition to run the analysis (column name in expression file)', required=True)
        parser.add_argument('-log_transform', help='Log transform expression data', action='store_true')
        parser.add_argument('-experience_directory', type=str, help='Directory for storing results (default: ./experience)', default='./experience')
        # parser.add_argument('-iupac', help='Use the IUPAC extended genetic alphabet', action='store_true')
        parser.add_argument('-iupac', help=argparse.SUPPRESS, action='store_true')  # not working yet
        # parser.add_argument('-gap', help='Allow gap (N) in genetic alphabet', action='store_true')
        parser.add_argument('-gap', help=argparse.SUPPRESS, action='store_true')  # not working yet
        parser.add_argument('-cache', help='Store k-mer occurrences', action='store_true')
        parser.add_argument('-verbose', help='Print img for each k-mer', action='store_true')
        parser.add_argument('-kmer_max_length', type=int, help='Set maximal k-mer  length allowed (0 for None, default)', default=0)
        parser.add_argument('-nb_bins', type=int, help='Number of bins for segmenting sequences', default=13)
        parser.add_argument('-uniform_bins', help='Bins have all the same length (by default, a polynomial generator is used such that bins close to the alignment point are smaller than bins far from the align point; cf. publication)', action='store_true')
        parser.add_argument('-set_bins', help='set bins used for building lattices (ex: -50,10,0,10,50)', default='')
        parser.add_argument('-correlation_increase', type=float, help='Minimum correlation difference for continuing exploration (rho_new - rho_old, default: 0.01 [i.e. 1%%])', default=0.01)
        parser.add_argument('-correlation_increase_ratio', type=float, help='Minimum correlation difference as ratio of previous correlation ((rho_new-rho_old)/rho_old, default: 0.10 [ie: 10%%])', default=0.10)
        parser.add_argument('-correlation_min', type=float, help='Stop exploration when correlation passes below this value (default: 0)', default=0.0)
        # parser.add_argument('-init_lasso', help='perform LASSO on segmented roots to continue exploration with residual errors', action='store_true')
        parser.add_argument('-init_lasso', help=argparse.SUPPRESS, action='store_true')  # not working yet
        parser.add_argument('-final_lasso', help='Fit a linear model with the variables identified during exploration', action='store_true')
        parser.add_argument('-no_progress', help='Hide progress bar', action='store_true')
        parser.add_argument('-random_seed', type=int, help='Seed used for random generator and selection of the learning/test sets (default: 686185200). The same seed must be used when analysing the different conditions of a series to ensure that the same learning/test sets are used in all conditions', default=686185200)
        parser.add_argument('-nb_thread', type=int, help='Maximal number of thread (-1 for max, default)', default=-1)
        # parser.add_argument('-logistic', help='adjustment for exploration of binary data', action='store_true')
        parser.add_argument('-logistic', help=argparse.SUPPRESS, action='store_true')  # not working yet
        parser.add_argument('-time', help='Print time spent in each step', action='store_true')
        namespace = parser.parse_args(args)
        logging.info('PropertiesWrapper: ' + str(namespace))
        self.motif_min_size = 2
        with open(namespace.fasta_file, 'r') as infile:
            should_break = False
            seq = ''
            for line in infile.readlines():
                line = line.strip()
                if line.startswith('>'):
                    if should_break:
                        self.sequence_size = len(seq)
                        break
                    else:
                        should_break = True
                else:
                    seq += line
        logging.info('PropertiesWrapper: sequences size: ' + str(self.sequence_size))
        # self.index_file = namespace.index_file[:-4]
        self.expression_file = namespace.expression_file
        self.target_condition = namespace.target_condition
        self.fasta_file = namespace.fasta_file
        self.log_transform = namespace.log_transform
        self.experience_directory = namespace.experience_directory
        if self.experience_directory.endswith('/'):
            self.experience_directory = self.experience_directory[:-1]
        self.alignement_point_index = namespace.alignement_point_index
        self.nb_thread = namespace.nb_thread
        if self.nb_thread <= 0:
            self.nb_thread = os.cpu_count()
        self.use_iupac = namespace.iupac
        self.use_gap = namespace.gap
        self.cache = namespace.cache
        self.no_progress = namespace.no_progress
        self.verbose = namespace.verbose
        self.kmer_max_length = namespace.kmer_max_length
        self.nb_regions = namespace.nb_bins
        self.random_seed = namespace.random_seed
        self.correlation_increase = namespace.correlation_increase
        self.correlation_increase_ratio = namespace.correlation_increase_ratio
        self.correlation_min = namespace.correlation_min
        self.init_lasso = namespace.init_lasso
        self.final_lasso = namespace.final_lasso
        self.logistic = namespace.logistic
        self.regions_fixed_length = namespace.uniform_bins
        self.set_bins = namespace.set_bins
        self.time = namespace.time
