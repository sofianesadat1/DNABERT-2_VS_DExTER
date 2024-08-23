import random

from scipy.stats import spearmanr

import numpy as np
from sklearn.metrics import roc_auc_score

np.warnings.filterwarnings('ignore')


import PropertiesWrapper
import Segmentator


class Domain:

    def __init__(self, kmer_str, rank, index):
        self.kmer_str = kmer_str
        self.len_kmer_str = len(self.kmer_str)
        self.rank = rank
        self.index = index
        self.is_root = False
        self.is_valid = False
        self.power = -1  # unused value ???? should I delete it ?
        self.parents = []
        self.controls = []

    def __eq__(self, other):
        return self.kmer_str == other.kmer_str and self.rank == other.rank and self.index == other.index

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.kmer_str, self.rank, self.index))

    def get_score(self, lattice_kmer):
        if lattice_kmer is None:
            return 0
        self.power = lattice_kmer.structure[self.rank][self.index]
        return self.power

    # def get_best_children(self, lattice_kmer):
    #     self.get_score(lattice_kmer)
    #     ret = []
    #     line = self.rank - 1
    #     last_column = self.index + 1
    #     best_found_score = 0
    #     while line >= 0:
    #         column = self.index
    #         while column <= last_column:
    #             if lattice_kmer.structure[line][column] > self.power:
    #                 tmp = Domain(self.kmer_str, line, column)
    #                 tmp.power = lattice_kmer.structure[line][column]
    #                 best_found_score = max(tmp.power, best_found_score)
    #                 ret.append(tmp)
    #             column += 1
    #         last_column += 1
    #         line -= 1
    #     ret = [o for o in ret if o.power == best_found_score]
    #     if len(ret) > 1:
    #         ret = [random.choice(ret)]
    #     return ret

    def get_domain_name(self, lattice_segments):
        left_bound, right_bound = lattice_segments.get_bounds(self.rank, self.index)
        return self.kmer_str + ' ' + str(left_bound) + ' ' + str(right_bound)

    def intersect(self, other):
        ret = None
        min1 = self.index
        max1 = self.index + self.rank
        min2 = other.index
        max2 = other.index + other.rank
        x = range(min1, max1 + 1)
        y = range(min2, max2 + 1)
        xs = set(x)
        ret = len(xs.intersection(y)) > 0
        return ret

    def get_graph_name(self, lattice_segments):
        left_bound, right_bound = lattice_segments.get_bounds(self.rank, self.index)
        return '\%' + self.kmer_str + '[' + str(left_bound) + ';' + str(right_bound) + '] rho: ' + '{0:.3f}'.format(self.power*100) + '\%'

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return '%' + self.kmer_str + '[' + str(self.rank) + ';' + str(self.index) + '] rho: ' + '{0:.3f}'.format(self.power*100) + '%'


class Lattice:

    def __init__(self):
        self.structure = []

    def __str__(self):
        max_e = -1
        for line in range(0, len(self.structure)):
            for column in range(0, len(self.structure[line])):
                tmp = '[' + str(self.structure[line][column]) + ']'
                max_e = max(len(tmp), max_e)
        ret = ''
        first_line = True
        max_l = -1
        for line in range(0, len(self.structure)):
            tmp = ''
            for column in range(0, len(self.structure[line])):
                tmp += ('[' + str(self.structure[line][column]) + ']').center(max_e)
            tmp = tmp.strip()
            if first_line:
                first_line = False
                max_l = len(tmp)
            tmp = tmp.center(max_l)
            ret = tmp + '\n' + ret
        return ret


class LatticeSequence(Lattice):

    def __init__(self, sequence, occurrences):
        super().__init__()
        lattice_segments = Segmentator.lattice_segments
        self.sequence = sequence
        self.structure = []
        for i in reversed(range(0, len(lattice_segments.structure[0]))):
            self.structure.append([0]*(i+1))
        for occurrence in occurrences:
            for index in range(len(lattice_segments.structure[0])):
                bounds = lattice_segments.structure[0][index]
                left_bound = bounds[0]
                right_bound = bounds[1]
                if left_bound <= occurrence <= right_bound:
                    self.structure[0][index] += 1
        for column in range(0, len(self.structure[0])-1):
            self.structure[1][column] = self.structure[0][column] + self.structure[0][column+1]
        for line in range(2, len(self.structure[0])):
            for column in range(0, len(self.structure[0])-line):
                self.structure[line][column] = self.structure[line-1][column] + self.structure[line-1][column+1] - self.structure[line-2][column+1]


class LatticeKmer(Lattice):

    def __init__(self, kmer, lst_lattice_sequences, dct_expr):
        super().__init__()
        lattice_segments = Segmentator.lattice_segments
        properties_wrapper = PropertiesWrapper.properties_wrapper_instance
        self.kmer = kmer
        self.structure = []
        for i in reversed(range(0, len(lattice_segments.structure[0]))):
            self.structure.append([0]*(i+1))
        for i in range(len(self.structure)):
            for j in range(len(self.structure[i])):
                missing_training = set(properties_wrapper.training_set)
                tmpX = []
                tmpY = []
                for lattice_sequence in lst_lattice_sequences:
                    tmpX.append(lattice_sequence.structure[i][j])
                    tmpY.append(dct_expr[lattice_sequence.sequence])
                    missing_training.remove(lattice_sequence.sequence)
                for missing in missing_training:
                    tmpX.append(0)
                    tmpY.append(dct_expr[missing])
                try:
                    if not properties_wrapper.logistic:
                        tmp = spearmanr(tmpX, tmpY)[0]
                        if np.isnan(tmp):
                            tmp = 0
                    else:
                        tmp = abs(roc_auc_score(tmpY, tmpX) - 0.5) + 0.5
                    self.structure[i][j] = abs(tmp)
                except:
                    pass

    def __str__(self):
        max_e = -1
        for line in range(0, len(self.structure)):
            for column in range(0, len(self.structure[line])):
                tmp = '[' + '{0:.3f}'.format(self.structure[line][column]) + ']'
                max_e = max(len(tmp), max_e)
        ret = ''
        first_line = True
        max_l = -1
        for line in range(0, len(self.structure)):
            tmp = ''
            for column in range(0, len(self.structure[line])):
                tmp += ('[' + '{0:.3f}'.format(self.structure[line][column]*100) + '%]').center(max_e)
            tmp = tmp.strip()
            if first_line:
                first_line = False
                max_l = len(tmp)
            tmp = tmp.center(max_l)
            ret = tmp + '\n' + ret
        return ret
