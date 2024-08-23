import logging
import os
import pickle
from collections import defaultdict
from datetime import datetime
from operator import itemgetter
from threading import Thread

import re

import Broker
import LatticePrinter
import build_model
import lasso
from GraphPrinter import GraphPrinter
import PropertiesWrapper
from multiprocessing.pool import Pool

import Segmentator
from Models import LatticeSequence, LatticeKmer, Domain

IUPAC = {
    'A': ['A'],
    'T': ['T'],
    'G': ['G'],
    'C': ['C'],
    'R': ['A', 'G'],
    'Y': ['T', 'C'],
    'M': ['A', 'C'],
    'K': ['T', 'G'],
    'S': ['G', 'C'],
    'W': ['A', 'T'],
    'H': ['A', 'T', 'C'],
    'B': ['T', 'G', 'C'],
    'V': ['A', 'G', 'C'],
    'D': ['A', 'T', 'G'],
    'N': ['A', 'T', 'G', 'C']
}


class JobScheduler(Thread):

    def __init__(self, broker, max_running_jobs):
        super().__init__()
        self.terminated = False
        self.ts = 0.0
        self.broker = broker
        JobScheduler.BROKER = broker
        self.max_running_jobs = max_running_jobs
        self.properties_wrapper = PropertiesWrapper.properties_wrapper_instance
        self.kmer_min_size = None
        self.pool = None
        self.pid = None
        self.lasso_performed = False

    def run(self):
        self.pid = int(os.popen('echo $PPID', 'r').read())
        for domain in self.broker.domain_obj_waiting:
            domain.get_score(self.treat_kmer(domain.kmer_str))
            self.properties_wrapper.graph_printer.print_root_domain(domain.get_graph_name(Segmentator.lattice_segments))
        while self.terminated is False:
            self.kmer_min_size = None
            for domain in self.broker.domain_obj_waiting:
                if not self.kmer_min_size:
                    self.kmer_min_size = domain.len_kmer_str
                self.kmer_min_size = min(self.kmer_min_size, domain.len_kmer_str)
            this_round_domains = [d for d in self.broker.domain_obj_waiting if d.len_kmer_str == self.kmer_min_size]
            this_round_domain = this_round_domains[0]

            if self.properties_wrapper.init_lasso:
                if not self.lasso_performed and this_round_domain.len_kmer_str > 2:
                    # print segmented roots
                    lst_root_domains_obj = []
                    for domain in self.broker.domain_obj_done:
                        if domain.is_root:
                            lst_root_domains_obj.append(domain)
                    with open(self.properties_wrapper.experience_directory + '/segmented_root_domains.dat', 'w') as outfile:
                        for domain in self.broker.domain_obj_done:
                            good_size = domain.len_kmer_str == 2
                            has_a_root_parent = False
                            for parent in domain.parents:
                                if parent in lst_root_domains_obj:
                                    has_a_root_parent = True
                            if domain.is_valid and has_a_root_parent and good_size:
                                outfile.write(domain.get_domain_name(Segmentator.lattice_segments) + '\n')
                    build_model.run(self.properties_wrapper.experience_directory + '/occurrences/', self.properties_wrapper.experience_directory + '/segmented_root_domains.dat', self.properties_wrapper.experience_directory + '/data/' + self.properties_wrapper.target_condition + '.data', self.properties_wrapper.training_set, self.properties_wrapper.experience_directory + '/models/segmented_root_domains.dat_training_set.log.matrix', self.properties_wrapper.alignement_point_index)
                    build_model.run(self.properties_wrapper.experience_directory + '/occurrences/', self.properties_wrapper.experience_directory + '/segmented_root_domains.dat', self.properties_wrapper.experience_directory + '/data/' + self.properties_wrapper.target_condition + '.data', self.properties_wrapper.testing_set, self.properties_wrapper.experience_directory + '/models/segmented_root_domains.dat_testing_set.log.matrix', self.properties_wrapper.alignement_point_index)
                    _1, _2, _3, _4, _5, dct_errors = lasso.run(self.properties_wrapper.experience_directory + '/models/segmented_root_domains.dat_training_set.log.matrix', self.properties_wrapper.experience_directory + '/models/segmented_root_domains.dat_testing_set.log.matrix')
                    self.properties_wrapper.dct_lasso_residual_error = dct_errors
                    self.lasso_performed = True

            new_domains = []
            new_domains.extend(self.treat_domain(this_round_domain))
            self.broker.domain_obj_waiting.remove(this_round_domain)
            self.broker.domain_obj_done.add(this_round_domain)

            for nd in new_domains:
                if nd not in self.broker.domain_obj_waiting and nd not in self.broker.domain_obj_done:
                    self.broker.domain_obj_waiting.append(nd)

            if len(self.broker.domain_obj_waiting) == 0:
                self.terminated = True
            # time.sleep(self.ts)
        self.pid = None

    def treat_kmer(self, kmer_str):
        if len(kmer_str) < self.properties_wrapper.motif_min_size:
            return None
        if kmer_str in self.broker.lattice_kmer_obj_ready:
            return self.broker.lattice_kmer_obj_ready[kmer_str]
        occurrences_file = search_occurrence(kmer_str)
        time_start = datetime.now()
        all_lattice_sequences = []
        dct_occurrences = pickle.load(open(occurrences_file, 'rb'))
        for seq_name, occurrences in dct_occurrences.items():
            tmp = treat_occurrences(seq_name, occurrences)
            if tmp is not None:
                all_lattice_sequences.append(tmp)
        # with open(occurrences_file, 'r') as infile:
        #     try:
        #         self.pool = Pool(processes=self.properties_wrapper.nb_thread)
        #         for lattice_sequence in self.pool.imap(treat_occ_line, infile.readlines()):
        #             if lattice_sequence is not None:
        #                 all_lattice_sequences.append(lattice_sequence)
        #         self.pool.close()
        #     except Exception as e:
        #         self.pool.close()
        #         raise e
        if self.properties_wrapper.init_lasso:
            if len(kmer_str) > 2:
                lattice_kmer = LatticeKmer(kmer_str, all_lattice_sequences, self.properties_wrapper.dct_lasso_residual_error)
            else:
                lattice_kmer = LatticeKmer(kmer_str, all_lattice_sequences, self.properties_wrapper.dct_gene_expr)
        else:
            lattice_kmer = LatticeKmer(kmer_str, all_lattice_sequences, self.properties_wrapper.dct_gene_expr)
        self.broker.lattice_kmer_obj_ready[kmer_str] = lattice_kmer
        if self.properties_wrapper.verbose:
            od = self.properties_wrapper.experience_directory + '/img/' + str(len(kmer_str))
            os.makedirs(od, exist_ok=True)
            with open(od + '/' + kmer_str + '.lattice', 'w') as outfile:
                outfile.write(str(lattice_kmer))
            if self.properties_wrapper.logistic:
                LatticePrinter.run(self.properties_wrapper.experience_directory + '/img/segments.lattice', od + '/' + kmer_str + '.lattice', 50, 75)
            else:
                LatticePrinter.run(self.properties_wrapper.experience_directory + '/img/segments.lattice', od + '/' + kmer_str + '.lattice', 0, 60)
        time_stop = datetime.now()
        self.broker.time_domains += (time_stop - time_start).total_seconds()
        return lattice_kmer

    def treat_domain(self, domain):
        ret = []
        is_valid = True
        if self.properties_wrapper.init_lasso and domain.len_kmer_str == 3 and domain.parents[0].len_kmer_str == 2:
            is_valid = domain.get_score(domain.kmer_str) > 0
        else:
            for parent in domain.parents:
                a = domain.get_score(self.treat_kmer(domain.kmer_str))
                b = parent.get_score(self.treat_kmer(parent.kmer_str))
                if domain.kmer_str != parent.kmer_str:
                    if b == 0:
                        if a < self.properties_wrapper.correlation_increase:
                            logging.info('JobScheduler.py: ' + str(domain) + ' has been rejected because of parent: ' + str(parent) + '.')
                            is_valid = False
                        else:
                            pass
                    elif (a - b) / b < self.properties_wrapper.correlation_increase_ratio or (a - b) < self.properties_wrapper.correlation_increase:
                        logging.info('JobScheduler.py: ' + str(domain) + ' has been rejected because of parent: ' + str(parent) + '.')
                        is_valid = False
                else:
                    if b == 0:
                        if a < self.properties_wrapper.correlation_increase:
                            logging.info('JobScheduler.py: ' + str(domain) + ' has been rejected because of parent: ' + str(parent) + '.')
                            is_valid = False
                        else:
                            pass
                    elif (a - b) / b < self.properties_wrapper.correlation_increase_ratio or (a - b) < self.properties_wrapper.correlation_increase:
                        logging.info('JobScheduler.py: ' + str(domain) + ' has been rejected because of parent: ' + str(parent) + '.')
                        is_valid = False
            for control in domain.controls:
                a = domain.get_score(self.treat_kmer(domain.kmer_str))
                b = control.get_score(self.treat_kmer(control.kmer_str))
                if b == 0:
                    if a < self.properties_wrapper.correlation_increase:
                        logging.info('JobScheduler.py: ' + str(domain) + ' has been rejected because of parent: ' + str(control) + '.')
                        is_valid = False
                    else:
                        pass
                elif (a - b) / b < self.properties_wrapper.correlation_increase_ratio or (a - b) < self.properties_wrapper.correlation_increase:
                    logging.info('JobScheduler.py: ' + str(domain) + ' has been rejected because of control: ' + str(control) + '.')
                    is_valid = False
        if is_valid:
            logging.info('JobScheduler.py: ' + str(domain) + ' has been validated.')
            for parent in domain.parents:
                self.properties_wrapper.graph_printer.print_domain(parent.get_graph_name(Segmentator.lattice_segments), parent.power, domain.get_graph_name(Segmentator.lattice_segments), domain.power, GraphPrinter.BLUE)
            for control in domain.controls:
                self.properties_wrapper.graph_printer.print_domain(control.get_graph_name(Segmentator.lattice_segments), control.power, domain.get_graph_name(Segmentator.lattice_segments), domain.power, GraphPrinter.GREEN)
            domain.is_valid = True
            if domain.get_score(self.treat_kmer(domain.kmer_str)) < self.properties_wrapper.correlation_min:
                return ret
            lattice_kmer = self.treat_kmer(domain.kmer_str)
            validated_children = []
            putative_children = []
            for l in range(len(lattice_kmer.structure)):
                for c in range(len(lattice_kmer.structure[l])):
                    if lattice_kmer.structure[l][c] >= domain.power:
                        putative_children.append([l, c, lattice_kmer.structure[l][c]])
            def bubble_sort(alist):
                for passnum in range(len(alist)-1,0,-1):
                    for i in range(passnum):
                        # alist[i] = [l,c,pow]
                        if (alist[i][2] < alist[i+1][2]) \
                                or (alist[i][2] == alist[i+1][2] and alist[i][0] > alist[i+1][0]) \
                                or (alist[i][2] == alist[i+1][2] and alist[i][0] == alist[i+1][0] and alist[i][1] > alist[i+1][1]):
                            alist[i], alist[i+1] = alist[i+1], alist[i]
                return alist
            putative_children = bubble_sort(putative_children)
            for putative_child in putative_children:
                child = Domain(domain.kmer_str, putative_child[0], putative_child[1])
                child.parents.append(domain)
                child.controls.append(Domain(child.kmer_str[1:], child.rank, child.index))
                child.controls.append(Domain(child.kmer_str[:-1], child.rank, child.index))
                child.power = child.get_score(self.treat_kmer(child.kmer_str))
                acceptable = True
                for validated_child in validated_children:
                    if child.intersect(validated_child):
                        logging.info('JobScheduler.py: ' + str(domain) + ' -> Putative child ' + str(child) + ' is rejected because it intersects with ' + str(validated_child))
                        acceptable = False
                if acceptable:
                    a = child.get_score(self.treat_kmer(child.kmer_str))
                    for control in child.controls:
                        b = control.get_score(self.treat_kmer(control.kmer_str))
                        if b == 0:
                            if a == 0:
                                acceptable = False
                                logging.info('JobScheduler.py: ' + str(domain) + ' -> Putative child ' + str(child) + ' is rejected because its correlation is too weak')
                            else:
                                continue
                        elif (a - b) / b < self.properties_wrapper.correlation_increase_ratio or (a - b) < self.properties_wrapper.correlation_increase:
                            logging.info('JobScheduler.py: ' + str(domain) + ' -> Putative child ' + str(child) + ' is rejected because of control ' + str(control))
                            acceptable = False
                if acceptable:
                    child.is_valid = True
                    validated_children.append(child)
            if len(validated_children) > 0:
                logging.info('JobScheduler.py: Expanding ' + str(domain) + ' into: ' + str(validated_children))
            else:
                logging.info('JobScheduler.py: Can\'t expand ' + str(domain) + ' anymore.')
            for child in validated_children:
                if child != domain:
                    self.properties_wrapper.graph_printer.print_domain(domain.get_graph_name(Segmentator.lattice_segments), domain.power, child.get_graph_name(Segmentator.lattice_segments), child.power, GraphPrinter.MAGENTA)
                    self.broker.domain_obj_done.add(child)
                if self.properties_wrapper.kmer_max_length == 0 or child.len_kmer_str < self.properties_wrapper.kmer_max_length:  # allows to set kmer max length
                    for n in self.broker.NUCLEOTIDES + self.broker.IUPAC:
                        if not n.endswith('N'):
                            tmp1 = Domain(child.kmer_str + n, child.rank, child.index)
                            tmp1.parents.append(child)
                            tmp1.controls.append(Domain(tmp1.kmer_str[1:], child.rank, child.index))
                            tmp1.controls.append(Domain(tmp1.kmer_str[:-1], child.rank, child.index))
                            ret.append(tmp1)
                        if not n.startswith('N'):
                            tmp2 = Domain(n + child.kmer_str, child.rank, child.index)
                            tmp2.parents.append(child)
                            tmp2.controls.append(Domain(tmp2.kmer_str[1:], child.rank, child.index))
                            tmp2.controls.append(Domain(tmp2.kmer_str[:-1], child.rank, child.index))
                            ret.append(tmp2)
        return ret


def treat_occurrences(name, occurrences):
    ret = None
    if name in PropertiesWrapper.properties_wrapper_instance.training_set:
        ret = LatticeSequence(name, occurrences)
    return ret


# def treat_occ_line(line):
#     ret = None
#     line = line.strip()
#     if len(line) > 0 and '$' in line:
#         tline = re.split(r'\$', line)
#         if tline[0] in PropertiesWrapper.properties_wrapper_instance.training_set:
#             occs = re.split(r',', tline[1])
#             occs = [int(o) for o in occs]
#             ret = LatticeSequence(tline[0], occs)
#     return ret


def search_occurrence(kmer_str):
    time_start = datetime.now()
    properties_wrapper = PropertiesWrapper.properties_wrapper_instance
    broker = JobScheduler.BROKER
    output_dir = properties_wrapper.experience_directory + '/occurrences/' + str(len(kmer_str))
    os.makedirs(output_dir, exist_ok=True)

    output_file = output_dir + '/' + kmer_str
    if not os.path.exists(output_file):
        search_in_fasta(kmer_str, output_file)
        # search_in_index(properties_wrapper.index_file, kmer_str, output_dir, output_file)
    time_stop = datetime.now()
    broker.time_kmers += (time_stop - time_start).total_seconds()
    return output_file


def search_in_fasta(kmer_str, output_file):
    tmp = PropertiesWrapper.properties_wrapper_instance.fasta_parser.find_with_hash(kmer_str)
    pickle.dump(tmp, open(output_file, 'wb'))


# def search_in_index(index_file, kmer_str, output_dir, output_file):
#     dct_gene_occurrences = defaultdict(list)
#     expanded_motif = set(expand_motif(kmer_str))
#     try:
#         if not os.path.isdir(output_dir):
#             os.makedirs(output_dir)
#     except FileExistsError:
#         pass
#     logging.info('JobScheduler: searching occurrences of ' + kmer_str)
#     if 1 < len(expanded_motif):
#         tmp_files = []
#         known_files = []
#         for motif in expanded_motif:
#             if os.path.isfile(output_dir + '/' + motif):
#                 known_files.append(output_dir + '/' + motif)
#             else:
#                 tmp_file = output_dir + '/' + kmer_str + '.' + motif + '.tmp'
#                 logging.debug('searchInIndex ' + index_file + ' ' + motif + ' > ' + tmp_file)
#                 os.system('searchInIndex ' + index_file + ' ' + motif + ' > ' + tmp_file)
#                 tmp_files.append(tmp_file)
#         for file in tmp_files + known_files:
#             with open(file, 'r') as infile:
#                 for line in infile.readlines():
#                     if line.startswith('#'):
#                         continue
#                     tline = re.split(r'\$', line)
#                     gene_name = tline[0]
#                     occurrences = re.split(r',', tline[1])
#                     dct_gene_occurrences[gene_name].extend(occurrences)
#         with open(output_file, 'a') as outfile:
#             for gene in dct_gene_occurrences:
#                 occurrences = dct_gene_occurrences[gene]
#                 occurrences = [int(o) for o in occurrences]
#                 occurrences.sort()
#                 line = gene + '$'
#                 for occ in occurrences:
#                     line += str(str(occ) + ',')
#                 line = line[:-1] + '\n'
#                 outfile.write(line)
#         for file in tmp_files:
#             os.remove(file)
#     else:
#         logging.debug('searchInIndex ' + index_file + ' ' + kmer_str + ' >> ' + output_file)
#         os.system('searchInIndex ' + index_file + ' ' + kmer_str + ' >> ' + output_file)


def expand_motif(kmer_str):
    ret = ['']
    for nucleotide in kmer_str:
        rempl = IUPAC[nucleotide]
        lst = len(rempl) * ret
        for i in range(len(lst)):
            lst[i] += rempl[int(i/len(ret))]
        ret = lst
    return ret
