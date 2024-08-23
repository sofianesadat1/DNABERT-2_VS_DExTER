import os
import sys

import random

import re

import logging

TRAINING_SAMPLE_RATIO = 2/3


import os
import random
import re
import logging
import sys

def generate_sequences_sets(fasta_file, exprfile=None, base_dir='.', seed=686185200, num_genes=None):
    training_file = base_dir + '/training_set.log'
    testing_file = base_dir + '/testing_set.log'
    if os.path.isfile(training_file) and os.path.isfile(testing_file):
        training_set = []
        testing_set = []
        with open(training_file, 'r') as infile:
            for line in infile.readlines():
                line = line.strip()
                training_set.append(line)
        with open(testing_file, 'r') as infile:
            for line in infile.readlines():
                line = line.strip()
                testing_set.append(line)
        return training_set, testing_set
    else:
        duplicated_sequences = []
        seqs = []
        validated_seqs = []
        with open(fasta_file, 'r') as infile:
            for line in infile.readlines():
                line = line.strip()
                if line.startswith('>'):
                    if line[1:] in seqs:
                        logging.error('!!! found duplicated sequences in fasta !!! : ' + line[1:])
                        duplicated_sequences.append(line[1:])
                    else:
                        seqs.append(line[1:])
        if len(duplicated_sequences) > 0:
            print()
            print('!!! Found multiple sequences with the same name in the fasta !!!')
            print('Concerned sequences: ' + str(duplicated_sequences))
            print('Fix this before running DExTER.')
            print('Don\'t forget to regenerate the index files.')
            sys.exit(-1)
        if exprfile is not None:
            with open(exprfile, 'r') as infile:
                for line in infile.readlines():
                    line = line.strip()
                    if line.startswith('#') or line.startswith('01STAT:MAPPED') or line.startswith('02STAT:NORM_FACTOR'):
                        continue
                    tline = re.split(r'\s+', line)
                    if tline[0] in ['gene', '00Annotation', 'enhancer', 'sequence']:
                        continue
                    else:
                        if tline[0] in seqs:  # allows to ignore sequences if no expression available
                            validated_seqs.append(tline[0])
        else:  # skip checking if exprfile not specified
            validated_seqs = seqs
        
        if num_genes is not None:
            validated_seqs = random.sample(validated_seqs, min(num_genes, len(validated_seqs)))

        random.seed(seed)
        nb_sequences = len(validated_seqs)
        sample_size = int(nb_sequences * TRAINING_SAMPLE_RATIO)
        training_set = random.sample(validated_seqs, sample_size)
        testing_set = list(set(validated_seqs)-set(training_set))
        with open(training_file, 'w') as outfile:
            for s in training_set:
                outfile.write(s+'\n')
        with open(testing_file, 'w') as outfile:
            for s in testing_set:
                outfile.write(s+'\n')
        return training_set, testing_set



if __name__ == '__main__':
    fasta = sys.argv[1]
    generate_sequences_sets(fasta)
