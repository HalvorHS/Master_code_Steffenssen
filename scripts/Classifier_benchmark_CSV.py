import csv

import pandas as pd
import numpy as np
import os
import time
from tmu.preprocessing.DNA_data_pre import preprocess_genome_data
from tmu.models.classification.tsetlin_classifier_kmer import tsetlin_classifier_kmer
from tmu.models.classification.tsetlin_classifier import tsetlin_classifier
from tmu.models.classification.random_forrest_classifier import RFC_classifier
from tmu.models.classification.GC_content import GC_contents
from tmu.models.classification.CNN_classifier import CNN_classifier


def save_to_csv(results, csv_filename):
    df = pd.DataFrame(results, columns=["x_length", "Classifier Name", "Average Time (s)", "Std Time (s)", "Min Time (s)", "Max Time (s)", "Median Time (s)",
                                         "Average Accuracy", "Std Accuracy", "Min Accuracy", "Max Accuracy", "Median Accuracy"])
    df.to_csv(csv_filename, mode='w', index=False)




if __name__ == "__main__":
    main_folders = [
        r'C:\Users\hhste\OneDrive - Norwegian University of Life Sciences\Documents\skole\Master\Tsetlin_maskin\Tsetlin_code\data\all_genes',
        #r'C:\Users\hhste\OneDrive - Norwegian University of Life Sciences\Documents\skole\Master\Tsetlin_maskin\Tsetlin_code\data\Escherichia_other',
        #r'C:\Users\hhste\OneDrive - Norwegian University of Life Sciences\Documents\skole\Master\Tsetlin_maskin\Tsetlin_code\data\Actinomycetota_or_other_phyla',
        #r'C:\Users\hhste\OneDrive - Norwegian University of Life Sciences\Documents\skole\Master\Tsetlin_maskin\Tsetlin_code\data\prokaryote',
        #r'C:\Users\hhste\OneDrive - Norwegian University of Life Sciences\Documents\skole\Master\Tsetlin_maskin\Tsetlin_code\data\Serratia_other',
        #r'C:\Users\hhste\OneDrive - Norwegian University of Life Sciences\Documents\skole\Master\Tsetlin_maskin\Tsetlin_code\data\Methanosarcina_other_archaea'
    ]

    save_filename = 'processed_data_7_mer.csv'

    for main_folder in main_folders:
        save_path = os.path.join(main_folder, save_filename)

        subfolders = [folder for folder in os.listdir(main_folder) if
                      os.path.isdir(os.path.join(main_folder, folder))]

        x_lengths = [20, 100, 500, 1000, 2000]

        with open(save_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # Write header row
            writer.writerow(
                ["X Length", "Classifier", "Avg Time", "Std Time", "Min Time", "Max Time", "Median Time",
                 "Avg Accuracy", "Std Accuracy", "Min Accuracy", "Max Accuracy", "Median Accuracy"])

            for x_length in x_lengths:
                gene_data = preprocess_genome_data(main_folder, x_length=x_length, target_samples=500, force_new=True)

                gene_data_1_sequences = gene_data[subfolders[0]].iloc[:, 0].tolist()
                gene_data_2_sequences = gene_data[subfolders[1]].iloc[:, 0].tolist()

                for classifier_name, classifier_function in [
                    ("Tsetlin Classifier k-mer",
                     lambda: tsetlin_classifier_kmer(gene_data_1_sequences, gene_data_2_sequences, Clause=1000,
                                                     show_bar=True)),
                    #("Tsetlin Classifier direct",
                    # lambda: tsetlin_classifier(gene_data_1_sequences, gene_data_2_sequences, x_length=x_length,
                    #                            Clause=1000, show_bar=True)),
                    #("Random Forest Classifier",
                    # lambda: RFC_classifier(gene_data_1_sequences, gene_data_2_sequences, k_value=5)),
                    #("GC Content",
                    # lambda: GC_contents(gene_data_1_sequences, gene_data_2_sequences)),
                    #("CNN Classifier",
                    # lambda: CNN_classifier(gene_data_1_sequences, gene_data_2_sequences, x_length))
                    ]:

                    time_values = []
                    accuracy_values = []

                    for i in range(1):
                        start_time = time.time()
                        accuracy = classifier_function()
                        run_time = time.time() - start_time
                        time_values.append(run_time)
                        accuracy_values.append(accuracy)

                    # Calculate statistics for this classifier
                    avg_time = np.mean(time_values)
                    std_time = np.std(time_values)
                    min_time = np.min(time_values)
                    max_time = np.max(time_values)
                    median_time = np.median(time_values)

                    avg_accuracy = np.mean(accuracy_values)
                    std_accuracy = np.std(accuracy_values)
                    min_accuracy = np.min(accuracy_values)
                    max_accuracy = np.max(accuracy_values)
                    median_accuracy = np.median(accuracy_values)

                    # Write results row
                    writer.writerow(
                        [x_length, classifier_name, avg_time, std_time, min_time, max_time, median_time,
                         avg_accuracy, std_accuracy, min_accuracy, max_accuracy, median_accuracy])

        print("Results saved to:", save_path)