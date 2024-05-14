import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import numpy as np
from sklearn.preprocessing import OneHotEncoder

def extract_kmers(sequence, k):
    kmers = [sequence[i:i+k] for i in range(len(sequence) - k + 1)]
    return kmers

def encode_dna_sequences(sequences, k):
    encoder = OneHotEncoder()
    encoded_sequences = []
    max_length = max(len(sequence) for sequence in sequences)
    for sequence in sequences:
        kmers = extract_kmers(sequence, k)
        encoded_kmers = encoder.fit_transform(np.array(kmers).reshape(-1, 1)).toarray()
        pad_length = max_length - encoded_kmers.shape[0]
        padded_encoded_kmers = np.pad(encoded_kmers, ((0, pad_length), (0, 0)), mode='constant')
        encoded_sequences.append(padded_encoded_kmers)
    return np.array(encoded_sequences)

def RFC_classifi(contigs_prokaryote, contigs_eukaryote, k_value=5):
    # Encode DNA sequences
    prokaryote_encoded = encode_dna_sequences(contigs_prokaryote, k_value)
    eukaryote_encoded = encode_dna_sequences(contigs_eukaryote, k_value)

    # Combine the encoded sequences
    X = np.concatenate((prokaryote_encoded, eukaryote_encoded))
    y = np.concatenate((np.zeros(len(prokaryote_encoded)), np.ones(len(eukaryote_encoded))))

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest Classifier
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    # Predict on the test set
    y_pred = clf.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy