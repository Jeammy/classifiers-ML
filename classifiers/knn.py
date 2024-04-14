import numpy as np
from classifiers.classifieur import Classifier
from helper.utils import Utils


class Knn(Classifier):
    def __init__(self, **kwargs):
        """
        C'est un Initializer. 
        Vous pouvez passer d'autre paramètres au besoin,
        c'est à vous d'utiliser vos propres notations
        """
        self.k = kwargs.get('k', 1)

    def train(self, train, train_labels): #vous pouvez rajouter d'autres attributs au besoin
        """
        C'est la méthode qui va entrainer votre modèle,
        train est une matrice de type Numpy et de taille nxm, avec 
        n : le nombre d'exemple d'entrainement dans le dataset
        m : le mobre d'attribus (le nombre de caractéristiques)
        
        train_labels : est une matrice numpy de taille nx1
        
        vous pouvez rajouter d'autres arguments, il suffit juste de
        les expliquer en commentaire
        
        """
        self.data = train
        self.labels = train_labels
        self.classes = np.unique(train_labels)
        self.nbOfClasses = len(np.unique(train_labels))
        
    def predict(self, x):
        """
        Prédire la classe d'un exemple x donné en entrée
        exemple est de taille 1xm
        """
        distances = []
        for i, entry in enumerate(self.data):
            distance = Utils.euclideanDistance(x, entry)
            distanceWithLabel = (self.labels[i], distance)
            distances.append(distanceWithLabel)
        distances = np.array(distances)
        distances = distances[distances[:,1].argsort()]
        nearest_neighbors = distances[:self.k]
        classes, counts = np.unique(nearest_neighbors[:,0], return_counts=True)
        return int(classes[np.argmax(counts)])

        
    def evaluate(self, X, y):
        """
        c'est la méthode qui va evaluer votre modèle sur les données X
        l'argument X est une matrice de type Numpy et de taille nxm, avec 
        n : le nombre d'exemple de test dans le dataset
        m : le mobre d'attribus (le nombre de caractéristiques)

        y : est une matrice numpy de taille nx1

        vous pouvez rajouter d'autres arguments, il suffit juste de
        les expliquer en commentaire
        """
        predictions = np.array([], dtype = int)
        for test in X:
            prediction = self.predict(test)
            predictions = np.append(predictions, prediction)
        conf_matrix = Utils.generateConfusionMatrix(X, predictions, y, self.nbOfClasses)
        accuracy = Utils.calculAccuracy(conf_matrix, self.classes)
        precision = Utils.calculPrecision(conf_matrix, self.classes)
        recall = Utils.calculRecall(conf_matrix, self.classes)
        f1 = Utils.calculF1Score(precision, recall)
        results = {"confusion_matrix":conf_matrix, "accuracy":accuracy, "precision":precision, "recall":recall, "f1":f1}
        return results