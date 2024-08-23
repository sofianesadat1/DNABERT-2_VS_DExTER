import os
import re
import sys

import pickle

import numpy as np

from collections import defaultdict

from tqdm import tqdm

from fasta_parser import FastaParser


def defdctzero():
    return defaultdict(zero)


def zero():
    return 0


def load_domains(domains_file, align_point):
    domains = {}
    with open(domains_file) as infile:
        for line in infile.readlines():
            line = line.strip()
            tline = re.split(r'\s+', line)
            domains[tline[0]+'_'+str(int(tline[1]))+'-'+str(int(tline[2]))] = [int(tline[1])+align_point, int(tline[2])+align_point]
    return domains


def pre_run(occurrences_dir, domains_file):  # , index_file):
    needed_kmers = set()
    with open(domains_file, 'r') as infile:
        for line in infile.readlines():
            line = line.strip()
            """
            GG -1196 -685
            CG -2000 2000
            AA -2000 2000
            TA -1196 -126
            """
            tline = re.split(r'\s+', line)
            needed_kmers.add(tline[0])
    os.makedirs(occurrences_dir, exist_ok=True)
    for kmer in tqdm(needed_kmers):
        kmer_dir = occurrences_dir + '/' + str(len(kmer))
        os.makedirs(kmer_dir, exist_ok=True)
        output_file = kmer_dir + '/' + kmer
        if not os.path.exists(output_file):
            # search_in_index(index_file, kmer, kmer_dir, output_file)
            search_in_fasta(kmer, output_file)


def search_in_fasta(kmer_str, output_file):
    global fasta_parser
    tmp = fasta_parser.find_with_hash(kmer_str)
    pickle.dump(tmp, open(output_file, 'wb'))


# def search_in_index(index_file, kmer_str, output_dir, output_file):
#     dct_gene_occurrences = defaultdict(list)
#     expanded_motif = set(expand_motif(kmer_str))
#     try:
#         if not os.path.isdir(output_dir):
#             os.makedirs(output_dir)
#     except FileExistsError:
#         pass
#     if 1 < len(expanded_motif):
#         tmp_files = []
#         known_files = []
#         for motif in expanded_motif:
#             if os.path.isfile(output_dir + '/' + motif):
#                 known_files.append(output_dir + '/' + motif)
#             else:
#                 tmp_file = output_dir + '/' + kmer_str + '.' + motif + '.tmp'
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
#         os.system('searchInIndex ' + index_file + ' ' + kmer_str + ' >> ' + output_file)


# IUPAC = {
#     'A': ['A'],
#     'T': ['T'],
#     'G': ['G'],
#     'C': ['C'],
#     'R': ['A', 'G'],
#     'Y': ['T', 'C'],
#     'M': ['A', 'C'],
#     'K': ['T', 'G'],
#     'S': ['G', 'C'],
#     'W': ['A', 'T'],
#     'H': ['A', 'T', 'C'],
#     'B': ['T', 'G', 'C'],
#     'V': ['A', 'G', 'C'],
#     'D': ['A', 'T', 'G'],
#     'N': ['A', 'T', 'G', 'C']
# }
#
#
# def expand_motif(kmer_str):
#     ret = ['']
#     for nucleotide in kmer_str:
#         rempl = IUPAC[nucleotide]
#         lst = len(rempl) * ret
#         for i in range(len(lst)):
#             lst[i] += rempl[int(i/len(ret))]
#         ret = lst
#     return ret


def run(occurrences_dir, domains_file, expression_file, training_set, output_file, align_point):
    dct_domains = load_domains(domains_file, align_point)
    # print('load expressions')
    dct_expr = defaultdict(zero)
    seqs = []
    lst_domains = []
    with open(expression_file, 'r') as infile:
        for line in infile.readlines():
            line = line.strip()
            if line.startswith('gene'):
                continue
            else:
                tline = re.split(r'\s+', line)
                if tline[0] in training_set:
                    seqs.append(tline[0])
                    dct_expr[tline[0]] = tline[1]
    # print('parsing data')
    dct_seq_dom_freq = defaultdict(defdctzero)
    for kmer in tqdm(dct_domains):
        borne_inf = dct_domains[kmer][0]
        borne_sup = dct_domains[kmer][1]
        kmer = re.split(r'_', kmer)[0]
        max_occ = (borne_sup - borne_inf + 1)  # / len(kmer)
        domain = kmer + '.' + str(borne_inf-align_point) + '_' + str(borne_sup-align_point)
        lst_domains.append(domain)

        dct_occurrences = pickle.load(open(occurrences_dir + '/' + str(len(kmer)) + '/' + kmer, 'rb'))
        for sequence in dct_occurrences:
            if sequence in training_set:
                for occ in dct_occurrences[sequence]:
                    if borne_inf <= occ <= borne_sup:
                        dct_seq_dom_freq[sequence][domain] += 1 / max_occ

        # with open(occurrences_dir + '/' + str(len(kmer)) + '/' + kmer, 'r') as infile:
        #     for line in infile.readlines():
        #         line = line.strip()
        #         if line.startswith('#'):
        #             continue
        #         if len(line) > 0:
        #             tline1 = re.split(r'\$', line)
        #             tline2 = re.split(r',', tline1[1])
        #             sequence = tline1[0]
        #             if sequence in training_set:
        #                 for occ in tline2:
        #                     occ = int(occ)
        #                     if borne_inf <= occ <= borne_sup:
        #                         dct_seq_dom_freq[sequence][domain] += 1 / max_occ
    # print('printing results')
    with open(output_file, 'w') as outfile:
        outfile.write('sequence expression')
        lst_domains.sort()
        for dom in lst_domains:
            outfile.write(' ' + dom)
        outfile.write('\n')
        for seq in seqs:
            outfile.write(seq + ' ' + str(dct_expr[seq]))
            for dom in lst_domains:
                value = np.clip(dct_seq_dom_freq[seq][dom], 0, 1)
                outfile.write(' ' + str(value))
            outfile.write('\n')


if __name__ == '__main__':
    if len(sys.argv) != 8:
        print('Usage: python3 ./build_model.py ./occurrences_dir ./domains_file.dat ./data/condition.data ./fasta_file ./training_set.dat align_point ./output_file.matrix')
    occurrences_dir = sys.argv[1]
    domains_file = sys.argv[2]
    expr_file = sys.argv[3]
    fasta_file = sys.argv[4]
    seq_set = sys.argv[5]
    align_point = int(sys.argv[6])
    output_file = sys.argv[7]
    global fasta_parser
    fasta_parser = FastaParser(fasta_file, pre_comp_hash=False)
    seqs = []
    with open(seq_set) as infile:
        for line in infile.readlines():
            seqs.append(line.strip())
    pre_run(occurrences_dir, domains_file)  # , index_file)
    run(occurrences_dir, domains_file, expr_file, seqs, output_file, align_point)
