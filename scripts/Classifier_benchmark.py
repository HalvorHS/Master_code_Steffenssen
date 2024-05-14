import os
import time
from tmu.preprocessing.DNA_data_pre import preprocess_genome_data
from tmu.models.classification.tsetlin_classifier_kmer import tsetlin_classifier_kmer
from tmu.models.classification.tsetlin_classifier import tsetlin_classifier
from tmu.models.classification.random_forrest_classifier import RFC_classifier
from tmu.models.classification.GC_content import GC_contents
from tmu.models.classification.CNN_classifier import CNN_classifier
from tabulate import tabulate

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYCUDA_NO_WARN_X'] = '1'


if __name__ == "__main__":
    main_folder = (r'C:\Users\hhste\OneDrive - Norwegian University of Life '
                   r'Sciences\Documents\skole\Master\Tsetlin_maskin\Tsetlin_code\data\all_genes')



    save_filename = 'processed_data.pkl'
    save_path = os.path.join(main_folder, save_filename)
    load_path = save_path

    x_lengths = [500, 1000, 2000]
    results = []

    for x_length in x_lengths:
        gene_data = preprocess_genome_data(main_folder, x_length=x_length, target_samples=500,
                                           save_path=save_path, load_path=load_path, force_new=True)

        gene_data_1_sequences = gene_data['eukaryote'].iloc[:, 0].tolist()
        gene_data_2_sequences = gene_data['prokaryote'].iloc[:, 0].tolist()

        # Timer for Tsetlin Classifier
        start_time = time.time()
        tsetlin_accuracy_1000 = tsetlin_classifier(gene_data_1_sequences, gene_data_2_sequences, x_length, Clause=1000,
                                                   show_bar=True)
        tsetlin_time_1000 = time.time() - start_time

        # Timer for Tsetlin Classifier
        start_time = time.time()
        tsetlin_accuracy_10000 = tsetlin_classifier_kmer(gene_data_1_sequences, gene_data_2_sequences, Clause=1000,
                                                  show_bar=True)
        tsetlin_time_10000 = time.time() - start_time

        # Timer for Random Forest Classifier
        start_time = time.time()
        RFC_accuracy = RFC_classifier(gene_data_1_sequences, gene_data_2_sequences, k_value=5)
        RFC_time = time.time() - start_time

        # Timer for GC Content
        start_time = time.time()
        GC_accuracy = GC_contents(gene_data_1_sequences, gene_data_2_sequences)
        GC_time = time.time() - start_time

        # Timer for CNN Classifier
        start_time = time.time()
        CNN_accuracy = CNN_classifier(gene_data_1_sequences, gene_data_2_sequences, x_length)
        CNN_time = time.time() - start_time

        # Append the results for this x_length to the results list
        results.append(
            [x_length, tsetlin_accuracy_1000, tsetlin_time_1000, tsetlin_accuracy_10000, tsetlin_time_10000,
             RFC_accuracy, RFC_time, GC_accuracy, GC_time, CNN_accuracy, CNN_time])

    # Create a table with accuracy values, method names, and x_length
    table = [["x_length", "Method", "Accuracy", "Time (s)"]]

    for result in results:
        (x_length,tsetlin_accuracy_1000, tsetlin_time_1000, tsetlin_accuracy_10000, tsetlin_time_10000,
         RFC_accuracy, RFC_time, GC_accuracy, GC_time, CNN_accuracy, CNN_time) = result

        table.append([x_length, "Tsetlin Classifier 1000 clauses", tsetlin_accuracy_1000, tsetlin_time_1000])
        table.append([x_length, "Tsetlin Classifier 10000 clauses", tsetlin_accuracy_10000, tsetlin_time_10000])
        table.append([x_length, "Random Forest Classifier", RFC_accuracy, RFC_time])
        table.append([x_length, "GC Content", GC_accuracy, GC_time])
        table.append([x_length, "CNN Classifier", CNN_accuracy, CNN_time])

    # Print the table
    print(tabulate(table, headers="firstrow", tablefmt="grid"))
