import numpy as np
import sys
import helper.load_datasets as load_datasets
from classifiers.naiveBayes import NaiveBayes # importer la classe du classifieur bayesien
from classifiers.knn import Knn # importer la classe du Knn
#importer d'autres fichiers et classes si vous en avez développés
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from helper.utils import Utils


"""
C'est le fichier main duquel nous allons tout lancer
Vous allez dire en commentaire c'est quoi les paramètres que vous avez utilisés
En gros, vous allez :
1- Initialiser votre classifieur avec ses paramètres
2- Charger les datasets
3- Entrainer votre classifieur
4- Le tester

"""

# Initialisez vos paramètres
train_ratio = 0.8

# Initialisez/instanciez vos classifieurs avec leurs paramètres
naive_bayes_iris = NaiveBayes()
naive_bayes_wine = NaiveBayes()
naive_bayes_abalone = NaiveBayes()

sklearn_naive_bayes_iris = GaussianNB()
sklearn_naive_bayes_wine = GaussianNB()
# sklearn_naive_bayes_abalone = GaussianNB()

knn_iris = Knn(k=5)
knn_wine = Knn(k=5)
knn_abalone = Knn(k=5)

sklearn_knn_iris = KNeighborsClassifier(n_neighbors=5)
sklearn_knn_wine = KNeighborsClassifier(n_neighbors=5)
sklearn_knn_abalone = KNeighborsClassifier(n_neighbors=5)

# Charger/lire les datasets
iris_train, iris_train_labels, iris_test, iris_test_labels = load_datasets.load_iris_dataset(train_ratio)
wine_train, wine_train_labels, wine_test, wine_test_labels = load_datasets.load_wine_dataset(train_ratio)
abalone_train, abalone_train_labels, abalone_test, abalone_test_labels = load_datasets.load_abalone_dataset(train_ratio)

# Entrainez votre classifieur
naive_bayes_iris.train(iris_train, iris_train_labels)
naive_bayes_wine.train(wine_train, wine_train_labels)
naive_bayes_abalone.train(abalone_train, abalone_train_labels)

sklearn_naive_bayes_iris.fit(iris_train, iris_train_labels)
sklearn_naive_bayes_wine.fit(wine_train, wine_train_labels)
# sklearn_naive_bayes_abalone.fit(iris_train, iris_train_labels)

knn_iris.train(iris_train, iris_train_labels)
knn_wine.train(wine_train, wine_train_labels)
knn_abalone.train(abalone_train, abalone_train_labels)

sklearn_knn_iris.fit(iris_train, iris_train_labels)
sklearn_knn_wine.fit(wine_train, wine_train_labels)
sklearn_knn_abalone.fit(abalone_train, abalone_train_labels)

"""
Après avoir fait l'entrainement, évaluez votre modèle sur 
les données d'entrainement.
IMPORTANT : 
    Vous devez afficher ici avec la commande print() de python,
    - la matrice de confusion (confusion matrix)
    - l'accuracy
    - la précision (precision)
    - le rappel (recall)
    - le F1-score
"""

print("################################################################################################")
print("##                               Evaluation on train dataset                                  ##")
print("################################################################################################")

print("-----------------------------------------Iris train----------------------------------------------\n")
# naive bayes
trainResult = naive_bayes_iris.evaluate(iris_train, iris_train_labels)
sklearn_predictions = sklearn_naive_bayes_iris.predict(iris_train)
sk_results = {
    "confusion_matrix":confusion_matrix(iris_train_labels, sklearn_predictions), 
    "accuracy":accuracy_score(iris_train_labels, sklearn_predictions), 
    "precision":precision_score(iris_train_labels, sklearn_predictions, average="macro"), 
    "recall":recall_score(iris_train_labels, sklearn_predictions, average="macro"), 
    "f1":f1_score(iris_train_labels, sklearn_predictions, average="macro")
}
print("Train results for Naive Bayes with iris dataset: ")
Utils.printResults(trainResult)
print("Train results for sklearn Naive Bayes with iris dataset: ")
Utils.printResults(sk_results)

# knn
trainResult = knn_iris.evaluate(iris_train, iris_train_labels)
sklearn_predictions = sklearn_knn_iris.predict(iris_train)
sk_results = {
    "confusion_matrix":confusion_matrix(iris_train_labels, sklearn_predictions), 
    "accuracy":accuracy_score(iris_train_labels, sklearn_predictions), 
    "precision":precision_score(iris_train_labels, sklearn_predictions, average="macro"), 
    "recall":recall_score(iris_train_labels, sklearn_predictions, average="macro"), 
    "f1":f1_score(iris_train_labels, sklearn_predictions, average="macro")
}
print("Train results for knn with iris dataset: ")
Utils.printResults(trainResult)
print("Train results for sklearn knn with iris dataset: ")
Utils.printResults(sk_results)

print("-----------------------------------------wine train----------------------------------------------\n")
# naive bayes
trainResult = naive_bayes_wine.evaluate(wine_train, wine_train_labels)
sklearn_predictions = sklearn_naive_bayes_wine.predict(wine_train)
sk_results = {
    "confusion_matrix":confusion_matrix(wine_train_labels, sklearn_predictions), 
    "accuracy":accuracy_score(wine_train_labels, sklearn_predictions), 
    "precision":precision_score(wine_train_labels, sklearn_predictions, average="macro"), 
    "recall":recall_score(wine_train_labels, sklearn_predictions, average="macro"), 
    "f1":f1_score(wine_train_labels, sklearn_predictions, average="macro")
}
print("Train results for Naive Bayes with wine dataset: ")
Utils.printResults(trainResult)
print("Train results for sklearn Naive Bayes with wine dataset: ")
Utils.printResults(sk_results)

# knn
trainResult = knn_wine.evaluate(wine_train, wine_train_labels)
sklearn_predictions = sklearn_knn_wine.predict(wine_train)
sk_results = {
    "confusion_matrix":confusion_matrix(wine_train_labels, sklearn_predictions), 
    "accuracy":accuracy_score(wine_train_labels, sklearn_predictions), 
    "precision":precision_score(wine_train_labels, sklearn_predictions, average="macro"), 
    "recall":recall_score(wine_train_labels, sklearn_predictions, average="macro"), 
    "f1":f1_score(wine_train_labels, sklearn_predictions, average="macro")
}
print("Train results for knn with wine dataset: ")
Utils.printResults(trainResult)
print("Train results for sklearn knn with wine dataset: ")
Utils.printResults(sk_results)

print("-----------------------------------------Abalone train-------------------------------------------\n")
# naive bayes
trainResult = naive_bayes_abalone.evaluate(abalone_train, abalone_train_labels)
# sklearn_result = sklearn_naive_bayes_abalone.predict(abalone_train)
print("Train results for Naive Bayes with abalone dataset: ")
Utils.printResults(trainResult)

# knn
trainResult = knn_abalone.evaluate(abalone_train, abalone_train_labels)
sklearn_predictions = sklearn_knn_abalone.predict(abalone_train)
sk_results = {
    "confusion_matrix":confusion_matrix(abalone_train_labels, sklearn_predictions), 
    "accuracy":accuracy_score(abalone_train_labels, sklearn_predictions), 
    "precision":precision_score(abalone_train_labels, sklearn_predictions, average="macro"), 
    "recall":recall_score(abalone_train_labels, sklearn_predictions, average="macro"), 
    "f1":f1_score(abalone_train_labels, sklearn_predictions, average="macro")
}
print("Train results for knn with abalone dataset: ")
Utils.printResults(trainResult)
print("Train results for sklearn knn with abalone dataset: ")
Utils.printResults(sk_results)


# Tester votre classifieur
"""
Finalement, évaluez votre modèle sur les données de test.
IMPORTANT : 
    Vous devez afficher ici avec la commande print() de python,
    - la matrice de confusion (confusion matrix)
    - l'accuracy
    - la précision (precision)
    - le rappel (recall)
    - le F1-score
"""

print("################################################################################################")
print("##                               Evaluation on test dataset                                   ##")
print("################################################################################################\n")


print("-----------------------------------------Iris test----------------------------------------------\n")
# naive bayes
testResult = naive_bayes_iris.evaluate(iris_test, iris_test_labels)
sklearn_predictions = sklearn_naive_bayes_iris.predict(iris_test)
sk_results = {
    "confusion_matrix":confusion_matrix(iris_test_labels, sklearn_predictions), 
    "accuracy":accuracy_score(iris_test_labels, sklearn_predictions), 
    "precision":precision_score(iris_test_labels, sklearn_predictions, average="macro"), 
    "recall":recall_score(iris_test_labels, sklearn_predictions, average="macro"), 
    "f1":f1_score(iris_test_labels, sklearn_predictions, average="macro")
}
print("Test results for Naive Bayes with iris dataset: ")
Utils.printResults(testResult)
print("Test results for sklearn Naive Bayes with iris dataset: ")
Utils.printResults(sk_results)

# knn
testResult = knn_iris.evaluate(iris_test, iris_test_labels)
sklearn_predictions = sklearn_knn_iris.predict(iris_test)
sk_results = {
    "confusion_matrix":confusion_matrix(iris_test_labels, sklearn_predictions), 
    "accuracy":accuracy_score(iris_test_labels, sklearn_predictions), 
    "precision":precision_score(iris_test_labels, sklearn_predictions, average="macro"), 
    "recall":recall_score(iris_test_labels, sklearn_predictions, average="macro"), 
    "f1":f1_score(iris_test_labels, sklearn_predictions, average="macro")
}
print("Test results for knn with iris dataset: ")
Utils.printResults(testResult)
print("Test results for sklearn knn with iris dataset: ")
Utils.printResults(sk_results)

print("-----------------------------------------wine test----------------------------------------------\n")
# naive bayes
testResult = naive_bayes_wine.evaluate(wine_test, wine_test_labels)
sklearn_predictions = sklearn_naive_bayes_wine.predict(wine_test)
sk_results = {
    "confusion_matrix":confusion_matrix(wine_test_labels, sklearn_predictions), 
    "accuracy":accuracy_score(wine_test_labels, sklearn_predictions), 
    "precision":precision_score(wine_test_labels, sklearn_predictions, average="macro"), 
    "recall":recall_score(wine_test_labels, sklearn_predictions, average="macro"), 
    "f1":f1_score(wine_test_labels, sklearn_predictions, average="macro")
}
print("Test results for Naive Bayes with wine dataset: ")
Utils.printResults(testResult)
print("Test results for sklearn Naive Bayes with wine dataset: ")
Utils.printResults(sk_results)

# knn
testResult = knn_wine.evaluate(wine_test, wine_test_labels)
sklearn_predictions = sklearn_knn_wine.predict(wine_test)
sk_results = {
    "confusion_matrix":confusion_matrix(wine_test_labels, sklearn_predictions), 
    "accuracy":accuracy_score(wine_test_labels, sklearn_predictions), 
    "precision":precision_score(wine_test_labels, sklearn_predictions, average="macro"), 
    "recall":recall_score(wine_test_labels, sklearn_predictions, average="macro"), 
    "f1":f1_score(wine_test_labels, sklearn_predictions, average="macro")
}
print("Test results for knn with wine dataset: ")
Utils.printResults(testResult)
print("Test results for sklearn knn with wine dataset: ")
Utils.printResults(sk_results)

print("-----------------------------------------Abalone test-------------------------------------------\n")
# abalone
testResult = naive_bayes_abalone.evaluate(abalone_test, abalone_test_labels)
# sklearn_result = sklearn_naive_bayes_abalone.predict(abalone_test)
print("Test results for Naive Bayes with abalone dataset: ")
Utils.printResults(testResult)

# knn
testResult = knn_abalone.evaluate(abalone_test, abalone_test_labels)
sklearn_predictions = sklearn_knn_abalone.predict(abalone_test)
sk_results = {
    "confusion_matrix":confusion_matrix(abalone_test_labels, sklearn_predictions), 
    "accuracy":accuracy_score(abalone_test_labels, sklearn_predictions), 
    "precision":precision_score(abalone_test_labels, sklearn_predictions, average="macro"), 
    "recall":recall_score(abalone_test_labels, sklearn_predictions, average="macro"), 
    "f1":f1_score(abalone_test_labels, sklearn_predictions, average="macro")
}
print("Test results for knn with abalone dataset: ")
Utils.printResults(testResult)
print("Test results for sklearn knn with abalone dataset: ")
Utils.printResults(sk_results)