# All rights reserved.
#
# This code is proprietary and confidential. Any unauthorized use, copying,
# modification, or distribution is strictly prohibited.
#
# Sachin Chhetri, [2024]

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC  # Support Vector Machine
from sklearn.metrics import accuracy_score
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

def load_data(directory):
    dirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    corpus = []
    labels = []
    files = []
    for dir in dirs:
        training = os.listdir(f'{directory}/{dir}')
        for book in training:
            files.append(book)
            labels.append(dir)
            all_words = ''
            with open(f'{directory}/{dir}/{book}', 'r', encoding='utf-8') as f:
                for line in f:
                    all_words += line.strip() + ' '
            corpus.append(all_words)
    return corpus, labels, files, dirs

def train_models(corpus, labels):
    vectorizer = TfidfVectorizer(stop_words='english', min_df=2, max_df=0.5)
    tfid_vect_results = vectorizer.fit_transform(corpus)

    # Training Naive Bayes
    nb_clf = MultinomialNB().fit(tfid_vect_results, labels)

    # Training SVM
    svm_clf = SVC(kernel='linear', probability=True).fit(tfid_vect_results, labels)

    # Training Logistic Regression
    lr_clf = LogisticRegression(max_iter=1000).fit(tfid_vect_results, labels)

    # Training Random Forest
    rf_clf = RandomForestClassifier(n_estimators=100).fit(tfid_vect_results, labels)

    return nb_clf, svm_clf, lr_clf, rf_clf, vectorizer

def predict_and_report(clf, vectorizer, new_directory, actual_labels=None):
    new_books = os.listdir(new_directory)
    all_predictions = []
    for book in new_books:
        all_words = ''
        with open(f'{new_directory}/{book}', 'r', encoding='utf-8') as f:
            for line in f:
                all_words += line.strip() + ' '
        tfid_vect_results = vectorizer.transform([all_words])
        prediction = clf.predict(tfid_vect_results)[0]
        all_predictions.append(prediction)
        print(f'For {book}, the predicted author is: {prediction}')

    # Only calculate accuracy if actual labels are provided and match the number of predictions
    if actual_labels and len(actual_labels) == len(all_predictions):
        accuracy = accuracy_score(actual_labels, all_predictions)
        print(f"Accuracy: {accuracy * 100:.2f}%")
    else:
        print("No accuracy calculation due to label mismatch or labels not provided.")

def plot_pca(tfid_vect_results, labels, dirs):
    pca = PCA(n_components=2)  # Reduce to two dimensions for visualization
    reduced_data = pca.fit_transform(tfid_vect_results.toarray())

    # Prepare labels array for colors
    unique_labels = list(set(labels))
    color_map = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))  # Generate a color map
    label_color = {label: color_map[i] for i, label in enumerate(unique_labels)}

    # Plot the PCA
    plt.figure(figsize=(10, 8))
    for label, color in label_color.items():
        condition = np.array(labels) == label
        plt.scatter(reduced_data[condition, 0], reduced_data[condition, 1], label=label, alpha=0.8, color=color)
    plt.title('PCA Visualization of TF-IDF data from 6 Authors')
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.grid(True)
    plt.savefig('Part2.png')
    plt.show()
def main():
    directory = 'TrainingNewBooks'
    new_directory = 'TestingBook'
    corpus, labels, files, dirs = load_data(directory)
    nb_clf, svm_clf, lr_clf, rf_clf, vectorizer = train_models(corpus, labels)

    print("Generating PCA visualization...")
    plot_pca(vectorizer.transform(corpus), labels, dirs)

    new_labels = ['Carroll', 'Grayson', 'Pocock', 'Shakespeare', 'Slesar', 'Snaith']

    print("\nNaive Bayes Predictions:")
    predict_and_report(nb_clf, vectorizer, new_directory, new_labels)

    print("\nSVM Predictions:")
    predict_and_report(svm_clf, vectorizer, new_directory, new_labels)

    print("\nLogistic Regression Predictions:")
    predict_and_report(lr_clf, vectorizer, new_directory, new_labels)

    print("\nRandom Forest Predictions:")
    predict_and_report(rf_clf, vectorizer, new_directory, new_labels)

if __name__ == "__main__":
    main()

