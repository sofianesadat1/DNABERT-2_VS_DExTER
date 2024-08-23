import logging
import os
import sys
from datetime import datetime

import math

import re

import pickle
from multiprocessing.pool import Pool

from os import system

from tqdm import tqdm

import Segmentator
import build_model
from fasta_parser import FastaParser
import lasso
import preload
from Broker import Broker
from Displayer import Displayer
from GraphPrinter import GraphPrinter
from JobScheduler import JobScheduler
import PropertiesWrapper

from Models import Domain
from Segmentator import generate_segment_lattice, generate_segment_lattice_uniform, generate_segment_lattice_given_bins
from generate_sequences_sets import generate_sequences_sets


PIDS = []


def run(args):
    start_time = datetime.now()
    logging.info('Main.py: Starting the procedure')
    print_header()
    PIDS.append(int(os.popen('echo $PPID', 'r').read()))

    print('Processing arguments...')
    logging.info('Main.py: Processing arguments...')
    PropertiesWrapper.properties_wrapper_instance = PropertiesWrapper.PropertiesWrapper(args)

    global properties_wrapper
    properties_wrapper = PropertiesWrapper.properties_wrapper_instance

    print('Testing requirements...')
    logging.info('Main.py: Testing requirements')
    if test_requirements():
        global broker
        Broker.broker_instance = Broker()
        broker = Broker.broker_instance

        print('Clearing directories...')
        logging.info('Main.py: Setting up directories')
        if not properties_wrapper.cache:
            clear_directories()
        make_directories()
        properties_wrapper.graph_printer = GraphPrinter(properties_wrapper.experience_directory + '/exploration_graph')

        print('Generating sequences sets...')
        logging.info('Main.py: Generating sequences sets')
        training_set, testing_set = generate_sequences_sets(properties_wrapper.fasta_file, properties_wrapper.expression_file, properties_wrapper.experience_directory, properties_wrapper.random_seed)
        properties_wrapper.training_set = training_set
        properties_wrapper.testing_set = testing_set

        if len(training_set) == 0 or len(testing_set) == 0:
            print('\033[1;31m' + '!!! Error in matching sequences names between fasta and expression file !!! Early stopping, please fix it.' + '\033[0m')
            exit(-1)
        else:
            print('Using ' + str(len(training_set)) + ' sequences in training set and ' + str(len(testing_set)) + ' sequences in testing set.')
            logging.info('Main.py: Using ' + str(len(training_set)) + ' sequences in training set and ' + str(len(testing_set)) + ' sequences in testing set.')

        print('Extract expression data...')
        logging.info('Main.py: Extract expression data')
        generate_expression_file(training_set, testing_set)

        print('Running pre-load procedure...', flush=True)
        logging.info('Main.py: Running pre-load procedure')
        nucs = ['A', 'C', 'G', 'T']
        kmers = [n1+n2 for n1 in nucs for n2 in nucs]

        preload_start = datetime.now()
        
        global fasta_parser
        fasta_parser = FastaParser(properties_wrapper.fasta_file)
        properties_wrapper.fasta_parser = fasta_parser
        os.makedirs(properties_wrapper.experience_directory + '/occurrences/2/', exist_ok=True)
        for kmer in tqdm(kmers):
            tmp = properties_wrapper.fasta_parser.find_with_hash(kmer)
            pickle.dump(tmp, open(properties_wrapper.experience_directory + '/occurrences/2/' + kmer, 'wb'))
        preload_stop = datetime.now()
        broker.time_preload = (preload_stop - preload_start).total_seconds()

        global lattice_segments
        lattice_done = False
        if len(properties_wrapper.set_bins) > 0:
            bins = re.split(r',', properties_wrapper.set_bins)
            for b in bins:
                try:
                    _ = int(b)
                except Exception as e:
                    raise ValueError(str(b) + ' is not an integer and can\'t be used to defined bins')
            if len(bins) > 1:
                lattice_segments = generate_segment_lattice_given_bins(bins, properties_wrapper.aligement_point_index)
                lattice_done = True
        if properties_wrapper.regions_fixed_length and not lattice_done:
            lattice_segments = generate_segment_lattice_uniform(properties_wrapper.nb_regions, properties_wrapper.sequence_size, properties_wrapper.alignement_point_index)
        elif not properties_wrapper.regions_fixed_length and not lattice_done:
            lattice_segments = generate_segment_lattice(properties_wrapper.nb_regions, properties_wrapper.sequence_size, properties_wrapper.alignement_point_index)
        Segmentator.lattice_segments = lattice_segments
        if properties_wrapper.verbose:
            od = properties_wrapper.experience_directory + '/img/'
            os.makedirs(od, exist_ok=True)
            with open(od+'/segments.lattice', 'w') as outfile:
                outfile.write(str(lattice_segments))

        scheduler = JobScheduler(broker, properties_wrapper.nb_thread)
        displayer = Displayer(broker, properties_wrapper.no_progress)

        print('Exploring...')
        logging.info('Main.py: Exploring')
        try:
            print('Initialization of the broker...')
            logging.info('Main.py: Initiate first domains')
            for n1 in broker.NUCLEOTIDES:
                for n2 in broker.NUCLEOTIDES:
                    kmer = n1 + n2
                    broker.kmer_str_waiting.append(kmer)
                    broker.kmer_str_seen.append(kmer)
                    tmp = Domain(kmer, lattice_segments.get_max_rank(), 0)
                    tmp.is_root = True
                    tmp.is_valid = True
                    broker.domain_obj_waiting.append(tmp)

            logging.info('Main.py: start displayer')
            displayer.start()
            logging.info('Main.py: start scheduler')
            scheduler.start()
            PIDS.append(displayer.pid)
            PIDS.append(scheduler.pid)
            scheduler.join()
            logging.info('Main.py: scheduler is done')
            displayer.join()

            logging.info('Main.py: print root domains')
            lst_root_domains_obj = []
            with open(properties_wrapper.experience_directory + '/root_domains.dat', 'w') as outfile:
                for domain in broker.domain_obj_done:
                    if domain.is_root:
                        lst_root_domains_obj.append(domain)
                        outfile.write(domain.get_domain_name(lattice_segments) + '\n')
                        properties_wrapper.graph_printer.print_root_domain(domain.get_graph_name(lattice_segments))
            properties_wrapper.graph_printer.close()
            properties_wrapper.graph_printer.generate_graph()


            logging.info('Main.py: print all domains')
            with open(properties_wrapper.experience_directory + '/all_domains.dat', 'w') as outfile:
                for domain in broker.domain_obj_done:
                    if domain.is_valid:
                        outfile.write(domain.get_domain_name(lattice_segments) + '\n')


            print()
            matrices_start = datetime.now()
            print('Generation LASSO matrices...')
            logging.info('Main.py: Generate LASSO matrices')

            print(' -> ALL domains')
            logging.info('Main.py: -> all training')
            build_model.run(properties_wrapper.experience_directory + '/occurrences/', properties_wrapper.experience_directory + '/all_domains.dat', properties_wrapper.experience_directory + '/data/' + properties_wrapper.target_condition + '.data', training_set, properties_wrapper.experience_directory + '/models/all_domains.dat_training_set.log.matrix', properties_wrapper.alignement_point_index)
            logging.info('Main.py: -> all testing')
            build_model.run(properties_wrapper.experience_directory + '/occurrences/', properties_wrapper.experience_directory + '/all_domains.dat', properties_wrapper.experience_directory + '/data/' + properties_wrapper.target_condition + '.data', testing_set, properties_wrapper.experience_directory + '/models/all_domains.dat_testing_set.log.matrix', properties_wrapper.alignement_point_index)

            matrices_stop = datetime.now()
            broker.time_matrices = (matrices_stop - matrices_start).total_seconds()
            
            if properties_wrapper.final_lasso:
                print()
                print('Computing LASSO...')
                logging.info('Main.py: computing LASSO')

                def print_lasso_results(domains_name, nb_iter, mse, pearson_correlation, score_r2, pretty_model):
                    msg1 = 'LASSO on ' + domains_name + ' domains'
                    msg11 = '-> nb_iter = ' + str(nb_iter)
                    msg2 = '-> MSE = ' + str(mse)
                    msg3 = '-> Cor = ' + str(pearson_correlation)
                    msg4 = '-> R2 = ' + str(score_r2)
                    msg5 = '-> ' + pretty_model
                    logging.info('Main.py: ' + msg1)
                    logging.info('Main.py: ' + msg11)
                    logging.info('Main.py: ' + msg2)
                    logging.info('Main.py: ' + msg3)
                    logging.info('Main.py: ' + msg4)
                    logging.info('Main.py: ' + msg5)
                    print(msg1, msg11, msg3, msg5, sep='\n')
                    with open(properties_wrapper.experience_directory + '/' + domains_name + '.lasso', 'w') as lassofile:
                        lassofile.write(msg1 + '\n')
                        lassofile.write(msg11 + '\n')
                        lassofile.write(msg2 + '\n')
                        lassofile.write(msg3 + '\n')
                        lassofile.write(msg4 + '\n')
                        lassofile.write(msg5 + '\n')

              

                nb_iter, mse, pearson_correlation, score_r2, pretty_model, _ = lasso.run(
                    properties_wrapper.experience_directory + '/models/all_domains.dat_training_set.log.matrix',
                    properties_wrapper.experience_directory + '/models/all_domains.dat_testing_set.log.matrix')
                print_lasso_results('ALL', nb_iter, mse, pearson_correlation, score_r2, pretty_model)

        except KeyboardInterrupt:
            displayer.terminated = True
            displayer.join()
    print()
    print('Procedure completed.')
    logging.info('Main.py: procedure completed.')
    end_time = datetime.now()
    print()
    print('Duration: {}'.format(end_time - start_time))
    logging.info('Main.py: ' + 'Duration: {}'.format(end_time - start_time))
    broker.time_whole = (end_time - start_time).total_seconds()
    
    if properties_wrapper.time:
        
        broker.time_preload = int(broker.time_preload)
        broker.time_whole = int(broker.time_whole)
        broker.time_matrices = int(broker.time_matrices)
        broker.time_domains = int(broker.time_domains)
        broker.time_kmers = int(broker.time_kmers)
        print()
        print('Time:')
        print('Preload: ' + str(broker.time_preload) + 's')
        print('Searching k-mers: ' + str(broker.time_kmers) + 's')
        print('Computing lattices and correlations: ' + str(broker.time_domains) + 's')
        print('Generating matrices: ' + str(broker.time_matrices) + 's')
        print('Whole procedure: ' + str(broker.time_whole) + 's')
        print()
        
        with open(properties_wrapper.experience_directory + '/time.dat', 'w') as outfile:
            outfile.write('preload kmers domains matrices whole\n')
            outfile.write(str(broker.time_preload) + ' ' + str(broker.time_kmers) + ' ' + str(broker.time_domains) + ' ' + str(broker.time_matrices) + ' ' + str(broker.time_whole))
    
    for pid in PIDS:
        if pid is not None:
            pass


def print_header():
    title = '| DExTER - Christophe Menichelli, Vincent Guitard, Sophie Lèbre, Jose-Juan Lopez-Rubio, Charles-Henri Lecellier, Laurent Bréhélin (2020) |'
    print('-'*len(title))
    print(title)
    print('-'*len(title))
    print('contact: christophe.menichelli@lirmm.fr & laurent.brehelin@lirmm.fr')
    print(flush=True)


def test_requirements():
    ret = True
    from shutil import which
    utils = ['dot']
    for geralt in utils:
        if which(geralt) is None:
            print('Error: can\'t find executable: ' + geralt + '. Please refer to documentation.')
            # ret = False  # stop execution
    return ret


def clear_directories():
    global properties_wrapper
    system('rm -rf ' + properties_wrapper.experience_directory + ' 2> /dev/null')


def make_directories():
    global properties_wrapper
    os.makedirs(properties_wrapper.experience_directory, exist_ok=True)
    directories = ['img', 'data', 'occurrences', 'models']
    for directory in directories:
        os.makedirs(properties_wrapper.experience_directory+'/' + directory, exist_ok=True)


def generate_expression_file(training_set, testing_set):
    global properties_wrapper
    nill_val = 'NILL'
    min_val = None
    print_lines = []
    dct_gene_expr = {}
    valid_sequences_set = set(training_set + testing_set)
    with open(properties_wrapper.expression_file, 'r') as infile:
        condition_index = None
        for line in tqdm(infile.readlines(), unit='seq'):
            line = line.strip()
            if line.startswith('#') or line.startswith('//') or line.startswith('01STAT:MAPPED') or line.startswith('02STAT:NORM_FACTOR'):
                continue
            tline = re.split(r'\s+', line)
            for i in range(len(tline)):
                if tline[i] == properties_wrapper.target_condition:
                    condition_index = i
                    break
            if condition_index is None:
                raise Exception('Condition not found in expression file')
            else:
                if tline[0] in valid_sequences_set:
                    valid_sequences_set.remove(tline[0])
                    expr_value = float(tline[condition_index])
                    if properties_wrapper.log_transform:
                        if expr_value > 0:
                            expr_value = math.log(expr_value, 10)
                            if min_val is None:
                                min_val = expr_value
                            else:
                                min_val = min(expr_value, min_val)
                        else:
                            expr_value = nill_val
                    print_lines.append(tline[0] + ' ' + str(expr_value))
    with open(properties_wrapper.experience_directory + '/data/' + properties_wrapper.target_condition + '.data', 'w') as outfile:
        for line in print_lines:
            if nill_val is not None and nill_val in line:
                line = line.replace(nill_val, str(min_val))
            tline = re.split(r'\s+', line)
            dct_gene_expr[tline[0]] = float(tline[1])
            outfile.write(line + '\n')
    pickle.dump(dct_gene_expr, open(properties_wrapper.experience_directory + '/data/' + properties_wrapper.target_condition + '.data.bin', 'wb'))
    properties_wrapper.dct_gene_expr = dct_gene_expr


def pre_load(kmer):
    global properties_wrapper
    preload.preload(properties_wrapper.fasta_file, properties_wrapper.experience_directory + '/occurrences/', kmer)


if __name__ == '__main__':
    logfile = str(datetime.now()) + '.log'
    logfile = logfile.replace(':', '_')
    logfile = logfile.replace(' ', '_')
    try:
        os.remove(logfile)
    except:
        pass
    logging.basicConfig(filename=logfile, level=logging.DEBUG, format='%(asctime)s %(levelname)-6s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', filemode='w')
    run(sys.argv[1:])
