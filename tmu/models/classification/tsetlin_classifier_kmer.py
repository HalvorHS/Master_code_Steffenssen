from tmu.preprocessing.DNA_data_pre import seq4int
import numpy as np
from math import sqrt, log
from tmu.models.classification.coalesced_classifier import TMCoalescedClassifier
from tmu.models.classification.vanilla_classifier import TMClassifier
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer


def tsetlin_classifier_kmer(gene_data_1, gene_data_2, Clause=1000, n_epochs=20, show_bar=True, seed=727):

    X = gene_data_1 + gene_data_2
    Y = [0] * len(gene_data_1) + [1] * len(gene_data_2)

    vectorizer = CountVectorizer(analyzer='char', ngram_range=(7, 7))
    X_features = vectorizer.fit_transform(X).toarray()

    binary_input = np.where(X_features > 0, 1, 0)

    T = round(sqrt(Clause/2) + 2)
    S = 2.543 * log(Clause/3.7579)

    best_accuracy = 0.0

    tm = TMClassifier(Clause, T, S, platform='CPU', weighted_clauses=True, type_iii_feedback=False)

    X_train, X_test, Y_train, Y_test = train_test_split(
        binary_input, Y, test_size=0.20, random_state=seed, stratify=Y)

    Y_train = np.array(Y_train)

    acc = np.zeros(n_epochs)
    if show_bar is True:
        for epochs in tqdm(range(n_epochs)):
            tm.fit(X_train, Y_train, patch_dim=(1, 5))
            Y_pred = tm.predict(X_test)
            acc[epochs] = np.mean(100 * (Y_pred == Y_test))
        accuracy = acc[-1]
        if accuracy > best_accuracy:
            best_accuracy = accuracy
    else:
        for epochs in range(n_epochs):
            tm.fit(X_train, Y_train, patch_dim=(1, 5))
            Y_pred = tm.predict(X_test)
            acc[epochs] = np.mean(100 * (Y_pred == Y_test))
        accuracy = acc[-1]
        if accuracy > best_accuracy:
            best_accuracy = accuracy

    return best_accuracy / 100