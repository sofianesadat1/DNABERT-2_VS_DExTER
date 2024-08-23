
#!/bin/bash

# Définissez le chemin vers votre projet
PATH_DEXTER=($pwd)

cd "${PATH_DEXTER}"

# Mise à jour de pip si nécessaire
echo "Mise à jour de pip..."
pip install --upgrade pip

# Vérifiez si bitarray est installé
if ! pip list | grep -F bitarray > /dev/null; then
    echo "Installation de bitarray..."
    pip install bitarray
else
    echo "bitarray est déjà installé."
fi

version=$(pip show numpy | grep Version | awk '{print $2}')
if [[ "$version" != "1.23.1" ]]; then
    pip uninstall numpy -y
    echo "Installation de numpy version spécifique..."
    pip install numpy==1.23.1
fi

# Vérification et installation de Graphviz si nécessaire
if ! dpkg -l | grep -qw graphviz; then
    echo "Mise à jour des paquets et installation de Graphviz..."
    sudo apt update
    sudo apt install graphviz -y
else
    echo "Graphviz est déjà installé."
fi



# ulimit -v 50000000  # Limite de 50 Go en kilo-octets
# Exécution du script principal
echo "Exécution du script principal..." && \
python3 ./Main.py -fasta_file ../DATASETS/GRCH38/2000_tss_2000/dexter/dexter_grch38_genes.fa \
    -expression_file ../DATASETS/GRCH38/2000_tss_2000/dexter/GTEx_gene_median_7tissues.tsv \
    -log_transform \
    -target_condition Pituitary \
    -alignement_point_index 2000 \
    -nb_bins 13 \
    -verbose \
    -experience_directory example/grch38 \
    -final_lasso \
    -time

