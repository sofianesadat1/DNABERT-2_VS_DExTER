import os
import re
import sys

colours = ['0000FF', '0020FF', '0040FF', '0061FF', '0081FF', '00A1FF', '00C2FF', '00E2FF', '00FFFA', '00FFDA', '00FFBA', '00FF99', '00FF79', '00FF59', '00FF38', '00FF18', '08FF00', '28FF00', '48FF00', '69FF00', '89FF00', 'AAFF00', 'CAFF00', 'EAFF00', 'FFF200', 'FFD200', 'FFB200', 'FF9100', 'FF7100', 'FF5000', 'FF3000', 'FF1000']


def run(regions_file, lattice_file, cmin=0, cmax=60):
    regions = open(regions_file, 'r').readlines()[-1]
    lattice_tab = open(lattice_file, 'r').readlines()
    base = os.path.basename(lattice_file)
    directory = lattice_file[:-len(base)]
    kmer_str = os.path.splitext(base)[0]
    __print_lattice__(regions, lattice_tab, directory, kmer_str, cmin, cmax)


def get_colour(value, colours, cmin, cmax):
    ret = (value - cmin) / (cmax - cmin)
    ret *= len(colours)
    ret = max(min(int(ret), len(colours) - 1), 0)
    return colours[ret]
    
    
def __print_lattice__(regions, lattice_tab, directory, kmer_str, cmin, cmax):
    tmp_file = directory + kmer_str + '.tmp'
    with open(tmp_file, 'w') as outfile:
        regions = re.findall(r'[-\d:]+', regions)
        max_len = max([len(x) for x in regions])

        outfile.write('strict digraph G {\n')
        outfile.write('    graph [fontname = "helvetica"];\n')
        outfile.write('    node [fontname = "helvetica"];\n')
        outfile.write('    edge [fontname = "helvetica"];\n')
        outfile.write('    graph [rankdir=BT];\n')
        outfile.write('    node [shape=record, color=lightblue2, style=filled];\n')
        outfile.write('    node [fixedsize=true, width=1.5, height=0.45];\n')

        current_line = 0
        current_column = 0
        regions_names = []
        for region in regions:
            regions_names.append(region.center(max_len))
            outfile.write('    "' + region.center(max_len) + '" [color="#90EE90", pos="' + str(current_line) + ',' + str(current_column) + '!"];\n')
            current_column += 1

        past_nodes_names = []
        current_line = 1
        current_id = 0
        for line in reversed(lattice_tab):
            current_column = 0
            current_nodes_names = []
            elements = re.findall(r'\d+\.\d+%', line)
            for element in elements:
                node_label = element.replace('%', '\%').center(max_len)
                colour = '#' + get_colour(float(element.replace('%', '')), colours, cmin, cmax)  # colours[max(min(int(float(element.replace('%', ''))*len(colours)/40), len(colours)-1),0)]
                outfile.write('    "' + str(current_id) + '" [label="' + node_label + '",pos="' + str(current_line) + ',' + str(current_column) + '!", style=filled, fillcolor="' + colour + '"];\n')
                current_nodes_names.append(str(current_id))
                if current_line == 1:
                    outfile.write('    "' + regions_names[current_column] + '" -> "' + str(current_id) + '";\n')
                else:
                    outfile.write('    "' + past_nodes_names[current_column] + '" -> "' + str(current_id) + '";\n')
                    outfile.write('    "' + past_nodes_names[current_column+1] + '" -> "' + str(current_id) + '";\n')
                current_id += 1
                current_column += 1
            past_nodes_names = current_nodes_names
            current_line += 1

        outfile.write('}')

    os.system('dot -T svg -o ' + directory + kmer_str + '.svg' + ' ' + tmp_file)
    os.remove(tmp_file)


if __name__ == '__main__':
    regions_file = sys.argv[1]
    lattice_file = sys.argv[2]
    if len(sys.argv) > 3:
        cmin = int(sys.argv[3])
        cmax = int(sys.argv[4])
    else:
        cmin = 0
        cmax = 60
    run(regions_file, lattice_file, cmin, cmax)
