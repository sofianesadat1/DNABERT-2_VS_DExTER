import os

from multiprocessing import Lock

import re

import PropertiesWrapper

graph_printer_instance = None


class GraphPrinter:

    BLUE = 'blue'
    GREEN = 'green'
    CYAN = 'cyan'
    MAGENTA = 'magenta'

    def __init__(self, file):
        self.svg_file = file + '.svg'
        self.graph_file = file + '.dat'
        self.outfile = open(self.graph_file, 'w')
        self.in_first = set()
        self.in_last = set()
        self.lock = Lock()
        self.initialize()

    def initialize(self):
        with self.lock:
            self.outfile.write('strict digraph G {\n')
            self.outfile.write('    labelloc="t";\n')
            if PropertiesWrapper.properties_wrapper_instance.init_lasso:
                self.outfile.write('    label="Exploration with init LASSO";\n')
            else:
                self.outfile.write('    label="Exploration";\n')
            self.outfile.write('    graph [fontname = "helvetica"];\n')
            self.outfile.write('    node [fontname = "helvetica"];\n')
            self.outfile.write('    edge [fontname = "helvetica"];\n')
            self.outfile.write('    graph [rankdir=LR]\n')
            self.outfile.write('    subgraph cluster_01 {\n')
            self.outfile.write('        graph [bgcolor=snow3]\n')
            self.outfile.write('        node [shape=plaintext]\n')
            self.outfile.write('        label = "Legend";\n')
            self.outfile.write('        "expansion" -> "   " [color=blue];\n')
            self.outfile.write('        "  control  " -> "  " [color=green];\n')
            self.outfile.write('        "segmentation" -> "    " [color=magenta];\n')
            self.outfile.write('    }\n')
            self.outfile.write('    subgraph cluster_02 {\n')
            self.outfile.write('        graph [bgcolor=snow3]\n')
            self.outfile.write('        node [shape=plaintext]\n')
            self.outfile.write('        label = "IUPAC extended codes";\n')
            self.outfile.write('        "R" -> "  A or G   ";\n')
            self.outfile.write('        "Y" -> "  T or C   ";\n')
            self.outfile.write('        "M" -> "  A or C   ";\n')
            self.outfile.write('        "K" -> "  G or T   ";\n')
            self.outfile.write('        "S" -> "  G or C   ";\n')
            self.outfile.write('        "W" -> "  A or T   ";\n')
            #self.outfile.write('        "H" -> "A or T or C";\n')
            #self.outfile.write('        "B" -> "T or C or G";\n')
            #self.outfile.write('        "V" -> "A or C or G";\n')
            #self.outfile.write('        "D" -> "A or T or G";\n')
            self.outfile.write('        "N" -> "    any    ";\n')
            self.outfile.write('    }\n')
            self.outfile.write('    subgraph cluster_03 {\n')
            self.outfile.write('        node [shape=record, color=lightblue2, style=filled]\n')
            self.outfile.flush()

    def print_root_domain(self, domain_name):
        with self.lock:
            self.outfile.write('        "' + domain_name + '";\n')
            tmp = re.split(r'\]', domain_name)[0].replace('[', ' ').replace(':', ' ')
            self.in_last.add(tmp[1:])
            self.outfile.flush()

    def print_domain(self, parent_name, parent_pred_power, domain_name, domain_pred_power, color):
        with self.lock:
            if parent_pred_power > -1:
                delta = '+INF'
                if parent_pred_power != 0:
                    delta = abs((domain_pred_power - parent_pred_power) / parent_pred_power) * 100
                    delta = '%.2f' % delta + '%'
                self.outfile.write('        edge [arrowsize=1, color=' + color + ']\n')
                self.outfile.write('        "' + parent_name + '"\t->\t"' + domain_name + '"[label="' + delta + '"];\n')
            tmp1 = re.split(r'\]', parent_name)[0].replace('[', ' ').replace(':', ' ')
            self.in_first.add(tmp1[1:])
            tmp2 = re.split(r'\]', domain_name)[0].replace('[', ' ').replace(':', ' ')
            self.in_last.add(tmp2[1:])
            self.outfile.flush()

    def close(self):
        with self.lock:
            self.outfile.write('}\n}\n')
            self.outfile.flush()
            self.outfile.close()

    def generate_graph(self):
        os.system('dot -T svg -o ' + self.svg_file + ' ' + self.graph_file)

    def get_all_domains(self):
        all_domains = self.in_last.union(self.in_first)
        return all_domains

    def get_leaf_domains(self):
        leaf = self.in_last - self.in_first
        return leaf
