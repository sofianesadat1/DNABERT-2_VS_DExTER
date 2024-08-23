import multiprocessing
import sys
import time

import bitarray
from collections import defaultdict, deque

from multiprocessing import Pool

from tqdm import tqdm


class BitsForgery:

    INSTANCE = None
    FORGERY = {}

    PRIME = 17  # la sardine
    LENGTH_ENCODING = 3
    TABLE = {'A': '000', 'C': '001', 'G': '010', 'T': '011', 'N': '100'}
    IUPAC = {'A': ['A'], 'C': ['C'], 'G': ['G'], 'T': ['T'], 'M': ['A', 'C'], 'R': ['A', 'G'], 'W': ['A', 'T'], 'S': ['C', 'G'], 'Y': ['C', 'T'], 'K': ['G', 'T'], 'B': ['C', 'G', 'T'], 'D': ['A', 'G', 'T'], 'H': ['A', 'C', 'T'], 'V': ['A', 'C', 'G'], 'N': ['A', 'C', 'G', 'T']}

    def __new__(cls, *args, **kwargs):
        if BitsForgery.INSTANCE is None:
            BitsForgery.INSTANCE = super(BitsForgery, cls).__new__(*args, **kwargs)
        return BitsForgery.INSTANCE

    @staticmethod
    def get(word):
        if word not in BitsForgery.FORGERY:
            words = BitsForgery.encode(word)
            lst = set()
            for w in words:
                lst.add(bitarray.frozenbitarray(w))
            lst = frozenset(lst)
            BitsForgery.FORGERY[word] = lst
        return BitsForgery.FORGERY[word]

    @staticmethod
    def get_ignore_iupac(word):
        if word not in BitsForgery.FORGERY:
            lst = set()
            lst.add(bitarray.frozenbitarray(''.join([BitsForgery.TABLE[_] for _ in word])))
            lst = frozenset(lst)
            BitsForgery.FORGERY[word] = lst
        return BitsForgery.FORGERY[word]

    @staticmethod
    def encode(word):
        ret = []
        for w in BitsForgery.develop_word(word):
            ret.append(''.join([BitsForgery.TABLE[_] for _ in w]))
        return ret

    @staticmethod
    def develop_word(word):
        ret = ['']
        for nucleotide in word:
            rempl = BitsForgery.IUPAC[nucleotide]
            lst = len(rempl) * ret
            for i in range(len(lst)):
                lst[i] += rempl[int(i / len(ret))]
            ret = lst
        return ret


class FastaParser:

    def __init__(self, fasta_file, pre_comp_hash=True):
        self.fasta = fasta_file
        lst_sequences = []
        self.dct_arrays = {}
        dct_sequences = defaultdict(list)
        with open(self.fasta, 'r') as infile:
            current_seq = ''
            for line in tqdm(infile):
                line = line.strip()
                if line.startswith('>'):
                    current_seq = line[1:]
                    lst_sequences.append(current_seq)
                else:
                    dct_sequences[current_seq].append(line.upper())
        for k in list(dct_sequences.keys()):
            self.dct_arrays[k] = next(iter(BitsForgery.get_ignore_iupac(''.join(dct_sequences[k]))))
            del dct_sequences[k]
        del dct_sequences
        self.dct_hashes = {}
        if pre_comp_hash:
            # print(' ...computing rolling hash k=2')
            self.__compute_rolling_hash__(2)
            # print(' ...computing rolling hash k=3')
            self.__compute_rolling_hash__(3)
            # print(' ...computing rolling hash k=4')
            self.__compute_rolling_hash__(4)

    @staticmethod
    def __compute_single_sequence__(items, length, len_encoding, prime):
        sequence_name = items[0]
        bits = items[1]
        seq_hash = []
        h = None
        deck = None
        for index in range(0, len(bits) - len_encoding * length, len_encoding):
            sub_sequence = bits[index:index + len_encoding * length]
            chunks = [sub_sequence[i * len_encoding:(i + 1) * len_encoding] for i in
                      range((len(sub_sequence) + len_encoding - 1) // len_encoding)]
            if h is None:
                deck = deque(chunks, len(chunks))
                h = 0
                for power in range(len(chunks)):
                    h += int(chunks[power].to01(), 2) * (prime ** power)
            else:
                old_chunck = deck.popleft()
                deck.append(chunks[-1])
                h = (h - int(old_chunck.to01(), 2)) / prime + int(deck[-1].to01(), 2) * (prime ** (len(chunks) - 1))
            seq_hash.append(h)
        return sequence_name, seq_hash


    
    def __compute_rolling_hash__(self, length):
        import PropertiesWrapper

        ret = {}
        nb_cpu = -1
        print("1 : " + str(nb_cpu))
        try:
            
            nb_cpu = PropertiesWrapper.properties_wrapper_instance.nb_thread
            print("2 : " + str(nb_cpu))
        except:
            nb_cpu = multiprocessing.cpu_count()
            print("3 : " + str(nb_cpu))
        with Pool(processes=nb_cpu) as pool:
            for sequence_name, seq_hash in pool.starmap(self.__compute_single_sequence__, zip(list(self.dct_arrays.items()), [length]*len(self.dct_arrays), [BitsForgery.LENGTH_ENCODING]*len(self.dct_arrays), [BitsForgery.PRIME]*len(self.dct_arrays)), chunksize=None):
                ret[sequence_name] = seq_hash
        self.dct_hashes[length] = ret
    

    def find(self, kmer):
        ret = defaultdict(list)
        tmp = BitsForgery.get(kmer)
        step = BitsForgery.LENGTH_ENCODING*len(kmer)
        for sequence_name in self.dct_arrays:
            sequence = self.dct_arrays[sequence_name]
            for i in range(0, len(sequence)-len(kmer), BitsForgery.LENGTH_ENCODING):
                if sequence[i:i+step] in tmp:
                    ret[sequence_name].append(int(i/3))
        return ret

    def find_list(self, lst_kmers):
        ret = defaultdict(lambda: defaultdict(list))
        tmp = [BitsForgery.get(kmer) for kmer in lst_kmers]
        step = BitsForgery.LENGTH_ENCODING*len(lst_kmers[0])
        for sequence_name in self.dct_arrays:
            sequence = self.dct_arrays[sequence_name]
            for i in range(0, len(sequence) - len(lst_kmers[0]), BitsForgery.LENGTH_ENCODING):
                sub_sequence = sequence[i:i+step]
                for kmer in tmp:
                    if sub_sequence in kmer:
                        ret[kmer][sequence_name].append(int(i/3))
        return ret

    def find_with_hash(self, kmer):
        ret = defaultdict(list)
        length = len(kmer)
        tmp = BitsForgery.get(kmer)
        possible_hashes = []
        for word in tmp:
            chunks = [word[i * BitsForgery.LENGTH_ENCODING:(i + 1) * BitsForgery.LENGTH_ENCODING] for i in range((len(word) + BitsForgery.LENGTH_ENCODING - 1) // BitsForgery.LENGTH_ENCODING)]
            h = 0
            for power in range(len(chunks)):
                h += int(chunks[power].to01(), 2) * (BitsForgery.PRIME ** power)
            possible_hashes.append(h)
        if length not in self.dct_hashes:
            self.__compute_rolling_hash__(length)
        for sequence_name in self.dct_hashes[length]:
            ret[sequence_name] = [index for index in range(len(self.dct_hashes[length][sequence_name])) if self.dct_hashes[length][sequence_name][index] in possible_hashes]
        return ret


if __name__ == '__main__':
    print('Initialization...')
    my_parser = FastaParser(sys.argv[1])
    print('done')
    nucs = ['A', 'C', 'G', 'T']
    total_old = 0
    total_hash = 0
    x = 0
    nucleotides = [n1+n2 for n1 in nucs for n2 in nucs]

    for n in nucleotides:
        start = time.time()
        my_parser.find(n)
        total_old += time.time() - start

        start = time.time()
        my_parser.find_with_hash(n)
        total_hash += time.time() - start
        x += 1
        print((str(x) + '/' + str(len(nucleotides)) + ' old: ' + str(total_old/x) + ' hashes: ' + str(total_hash/x)).center(120), end='\r')
    print()
