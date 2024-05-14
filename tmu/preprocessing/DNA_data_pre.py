import os
import pandas as pd
import random
import numpy as np
from Bio import SeqIO
import pickle

def argmaxarray(X, shape):
    inds = []
    cp = np.cumprod(shape)
    ind = np.argmax(X)
    ls = len(shape)
    for i in range(1, ls):
        m = ind // cp[ls-i-1]
        ind -= m*cp[ls-1-i]
        inds.append(m)
    inds.append(ind)
    return(inds)

def seq4int(seqs, max_len):
    n_seq = len(seqs)
    ints = np.zeros((n_seq, max_len*4), dtype=bool)
    for i in range(n_seq):
        seq_i = list(seqs[i])
        for j in range(min(max_len,len(seq_i))):
            if seq_i[j] == 'A':
                ints[i,(4*j)] = 1
            if seq_i[j] == 'C':
                ints[i,(4*j)+1] = 1
            if seq_i[j] == 'G':
                ints[i,(4*j)+2] = 1
            if seq_i[j] == 'T':
                ints[i,(4*j)+3] = 1
    return ints

def preprocess_genome_data(main_folder, x_length, target_samples=None, save_path=None, load_path=None, force_new=False):
    # Check if data should be loaded from a file
    random.seed(727)
    if not force_new and load_path is not None and os.path.isfile(load_path):
        with open(load_path, 'rb') as load_file:
            return pickle.load(load_file)

    # Initialize dictionary to store data for each folder
    folder_data = {}

    for folder_name in os.listdir(main_folder):
        folder_path = os.path.join(main_folder, folder_name)

        if os.path.isdir(folder_path):
            folder_data[folder_name] = []  # Initialize data list for this folder

            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)

                if filename.endswith('.fasta') or filename.endswith('.fna'):
                    with open(file_path, 'r') as file:
                        records = SeqIO.parse(file, 'fasta')

                        for record in records:
                            sequence = str(record.seq.upper())

                            # Skip sequences that are shorter than x_length
                            if len(sequence) < x_length:
                                continue

                            # Randomly sample subsequences
                            for _ in range(min(target_samples // 2, len(sequence) - x_length + 1)):
                                start_position = random.randint(0, len(sequence) - x_length)
                                subsequence = sequence[start_position:start_position + x_length]

                                folder_data[folder_name].append({'X': subsequence})

    # Ensure that all folder data has at least target_samples samples
    min_samples = target_samples if target_samples is not None else min(len(data) for data in folder_data.values())

    # Create separate dataframes for each folder's data
    folder_dfs = {}
    for folder_name, data in folder_data.items():
        folder_dfs[folder_name] = pd.DataFrame(data[:min_samples]).sample(frac=1, random_state=42).reset_index(drop=True)

    # Save the processed data if a save path is provided
    if save_path is not None:
        with open(save_path, 'wb') as save_file:
            pickle.dump(folder_dfs, save_file)

    return folder_dfs



if __name__ == "__main__":
    main_folder = (r'C:\Users\hhste\OneDrive - Norwegian University of Life '
                   r'Sciences\Documents\skole\Master\Tsetlin_maskin\Tsetlin_code\data')

    save_filename = 'processed_data.pkl'
    save_path = os.path.join(main_folder, save_filename)

    load_path = save_path  # Set to the path of the saved data if you want to load it

    x_length = 1000

    gene_data_pro, gene_data_euk = preprocess_genome_data(main_folder, x_length=x_length, target_samples=500,
                                                          save_path=save_path, load_path=load_path)

    GC_count_list_pro = []
    for contig in gene_data_pro['X']:
        GC_count = 0
        for letter in contig:
            if letter.upper() == 'C' or letter.upper() == 'G':
                GC_count += 1
        GC_count_list_pro.append(GC_count/len(contig))

    GC_percent = sum(GC_count_list_pro)/len(GC_count_list_pro)
    print('GC Percent pro: ', GC_percent)

    GC_count_list_euk = []
    for contig in gene_data_euk['X']:
        GC_count = 0
        for letter in contig:
            if letter.upper() == 'C' or letter.upper() == 'G':
                GC_count += 1
        GC_count_list_euk .append(GC_count / len(contig))

    GC_percent = sum(GC_count_list_euk ) / len(GC_count_list_euk )
    print('GC Percent euk: ', GC_percent)


'''
    gene_data_pro, gene_data_eur = preprocess_genome_data(main_folder, x_length=x_length, target_samples=500,
                                                          save_path=save_path, load_path=load_path)

    best_parameters, best_accuracy = run_code(gene_data_pro['X'], gene_data_eur['X'], x_length)

    print("Best Parameters:", best_parameters, "accuracy:", best_accuracy)
    

'''
