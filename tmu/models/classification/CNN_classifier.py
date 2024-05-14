import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import  GlobalMaxPooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D,  Dense
from tensorflow.keras.optimizers import Adam
from ann_visualizer.visualize import ann_viz


def CNN_classifier(gene_data_1, gene_data_2, max_sequence_length, seed=727):
    # Combine sequences and create labels
    sequences = gene_data_1 + gene_data_2
    labels = [0] * len(gene_data_1) + [1] * len(gene_data_2)

    # One-hot encode sequences
    sequences = np.array([[[1, 0, 0, 0] if char == 'T' else
                           [0, 1, 0, 0] if char == 'G' else
                           [0, 0, 1, 0] if char == 'C' else
                           [0, 0, 0, 1] for char in seq] for seq in sequences])

    labels = np.array(labels)

    # Use train_test_split with specified seed
    X_train, X_test, Y_train, Y_test = train_test_split(sequences, labels, test_size=0.2, random_state=seed)

    # Build the model
    model = Sequential([
        Conv1D(filters=128, kernel_size=7, activation='relu', padding='same', input_shape=(max_sequence_length, 4)),
        GlobalMaxPooling1D(),
        Dense(units=64, activation='relu'),
        Dense(units=1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    cnn_model = model.fit(X_train, Y_train, epochs=10, batch_size=32,
                          validation_data=(X_test, Y_test), verbose=0)


    # Get the final validation accuracy
    accuracy = cnn_model.history['val_accuracy'][-1]

    return accuracy


