import numpy as np
import os
import pandas as pd
def GC_contents(gene_data_1, gene_data_2, cutoff=0.5):
    def calculate_GC(sequence):
        CG_count = sum(letter.upper() in ['G', 'C'] for letter in sequence)
        return CG_count / len(sequence)

    prediction_1 = [1 if calculate_GC(contig) > cutoff else 0 for contig in gene_data_1]
    prediction_2 = [1 if calculate_GC(contig) > cutoff else 0 for contig in gene_data_2]

    all_predictions = np.hstack([prediction_1, prediction_2])

    Y = [0] * len(gene_data_1) + [1] * len(gene_data_2)

    # Calculate accuracy
    accuracy = np.mean(all_predictions == Y)
    if accuracy < cutoff:
        accuracy = 1 - accuracy
    return accuracy
