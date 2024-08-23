import os
from collections import defaultdict


def __open_preload__(fasta_file):
    global dct_fasta
    dct_fasta = defaultdict(list)
    # print('-reading fasta file')
    with open(fasta_file, 'r') as infile:
        current_gene = ''
        for line in infile.readlines():
            line = line.strip()
            if line.startswith('>'):
                current_gene = line[1:]
            else:
                dct_fasta[current_gene].append(line.upper())
    # print('-formatting data')
    for gene in dct_fasta:
        sequence = ''.join(dct_fasta[gene])
        dct_fasta[gene] = sequence


def __clean_preload__():
    global dct_fasta
    del dct_fasta


def dft():
    return defaultdict(list)


def preload(fasta_file, outdir, k_mer):
    if not os.path.exists(outdir + str(len(k_mer))):
        try:
            os.makedirs(outdir + str(len(k_mer)))
        except FileExistsError:
            pass
    if os.path.isfile(outdir + str(len(k_mer)) + '/' + k_mer):
        # print('-skipping', k_mer, 'file already exists.')
        return

    global dct_fasta
    try:
        if dct_fasta is None:
            pass
    except NameError:
        __open_preload__(fasta_file)

    # print('-searching occurrences of', k_mer)
    dct_kmer_gene_indexes = defaultdict(dft)
    for gene in dct_fasta:
        sequence = dct_fasta[gene]
        for index in range(len(sequence)):
            word = sequence[index:index+len(k_mer)]
            if k_mer == word:
                current_kmer = dct_kmer_gene_indexes[k_mer]
                current_kmer_gene = current_kmer[gene]
                current_kmer_gene.append(str(index))
                current_kmer_gene.append(',')
    # print('-printing results')
    for kmer in dct_kmer_gene_indexes:
        with open(outdir + str(len(kmer)) + '/' + kmer, 'w') as outfile:
            curr_dir = dct_kmer_gene_indexes[kmer]
            for gene in curr_dir:
                outfile.write(gene + '$')
                indexs = ''.join(curr_dir[gene])
                indexs = indexs[:-1]
                outfile.write(indexs + '\n')


if __name__ == '__main__':
    kmers = []
    for n1 in ['A', 'T', 'G', 'C']:
        for n2 in ['A', 'T', 'G', 'C']:
            for n3 in ['A', 'T', 'G', 'C', '']:
                kmers.append(n1+n2+n3)
    for kmer in kmers:
        preload('/home/drekkenov/regulation_domains_work/data/hg19.cage.maxpeak.minus5000.plus5000.coding_mRNA.fa', '/home/drekkenov/regulation_domains_work/occurrences/', kmer)
