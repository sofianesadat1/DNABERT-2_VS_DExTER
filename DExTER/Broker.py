import logging

import PropertiesWrapper


broker_instance = None


class Broker:

    def __init__(self):
        self.properties_wrapper = PropertiesWrapper.properties_wrapper_instance
        self.NUCLEOTIDES = ['A', 'T', 'G', 'C']
        self.IUPAC = []
        tmp = ['R', 'Y', 'M', 'K', 'S', 'W', 'H', 'B', 'V', 'D']
        if self.properties_wrapper.use_iupac:
            self.IUPAC.extend(tmp)
        anys = []
        for nuc in self.NUCLEOTIDES + self.IUPAC:
            anys.append('N' + nuc)
            anys.append(nuc + 'N')
        if self.properties_wrapper.use_gap:
            self.IUPAC.extend(anys)
        self.kmer_str_waiting = []
        self.kmer_str_seen = []
        self.lattice_kmer_obj_ready = {}

        self.domain_obj_waiting = []
        self.domain_obj_done = set()

        self.time_whole = 0
        self.time_kmers = 0
        self.time_domains = 0
        self.time_preload = 0
        self.time_matrices = 0
