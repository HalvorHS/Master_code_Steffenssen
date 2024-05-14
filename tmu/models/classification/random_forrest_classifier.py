import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer


def RFC_classifier(gene_data_1, gene_data_2, k_value=5, seed=727):

    X = gene_data_1 + gene_data_2
    Y = [0] * len(gene_data_1) + [1] * len(gene_data_2)

    vectorizer = CountVectorizer(analyzer='char', ngram_range=(k_value, k_value))
    X_features = vectorizer.fit_transform(X)

    X_train, X_test, Y_train, Y_test = train_test_split(X_features, Y, test_size=0.2, random_state=seed)
    clf = RandomForestClassifier()
    clf.fit(X_train, Y_train)

    Y_pred = clf.predict(X_test)

    accuracy = accuracy_score(Y_test, Y_pred)

    return accuracy
