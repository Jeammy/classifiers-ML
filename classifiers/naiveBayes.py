from math import sqrt
import numpy as np
from helper.utils import Utils

from classifiers.classifieur import Classifier

class NaiveBayes(Classifier):
    def __init__(self, **kwargs):
        """
        C'est un Initializer. 
        Vous pouvez passer d'autre paramètres au besoin,
        c'est à vous d'utiliser vos propres notations
        """
    

    def getNbOfClasses(self, array):
        """
        Renvoyer le nombre de classes
        """
        classes, count = np.unique(array, return_counts=True)
        return dict(zip(classes, count))
    

    ## get probability for a specific feature
    def calculProba(self, x, mean, std):
        exponent = -((x - mean) ** 2 / (2 * std ** 2))
        return (1 / (std * np.sqrt(2 * np.pi))) * np.exp(exponent)
    
    ## calculate probability for each class
    def classProbabilities(self, x):
        probByClass = dict()
        for i in range(len(self.nbOfClasses)):
            probByClass[i] = []
        for classe, data in self.summaries.items():
            probByClass[classe] = 1
            for column, columnData in enumerate(data):
                mean, std = self.summaries[classe][column]
                prob = self.calculProba(x[column], mean, std)
                probByClass[classe] *= prob.max()
        return probByClass


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
        nbColumns = train.shape[1]
        dataByClass = dict()
        self.nbOfClasses = self.getNbOfClasses(train_labels)
        self.summaries = dict()
        for i in range(len(self.nbOfClasses)):
            dataByClass[i] = []
            self.summaries[i] = []
        for i in range(len(train)):
            classe = dataByClass[train_labels[i]]
            row = train[i]
            classe.append(row)
            dataByClass[train_labels[i]] = classe
        for i in range(len(self.nbOfClasses)):
            dataByClass[i] = np.array(dataByClass[i])
        for key, classes in dataByClass.items():
            for i in range(nbColumns):
                column = classes[:,i]
                mean = Utils.mean(column)
                std = Utils.standardDeviation(column)
                self.summaries[key].append((mean, std))        
        return self.summaries
        
        
    def predict(self, x):
        """
        Prédire la classe d'un exemple x donné en entrée
        exemple est de taille 1xm
        """
        classProbalities = self.classProbabilities(x)
        bestClass = None
        bestProb = -1
        for classValue, prob in classProbalities.items():
            if bestClass is None or prob > bestProb:
                bestClass = classValue
                bestProb = prob
        return bestClass


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
        nbOfClasses = len(self.nbOfClasses)
        for test in X:
            prediction = self.predict(test)
            predictions = np.append(predictions, prediction)
        conf_matrix = Utils.generateConfusionMatrix(X, predictions, y, nbOfClasses)
        accuracy = Utils.calculAccuracy(conf_matrix, self.nbOfClasses)
        precision = Utils.calculPrecision(conf_matrix, self.nbOfClasses)
        recall = Utils.calculRecall(conf_matrix, self.nbOfClasses)
        f1 = Utils.calculF1Score(precision, recall)
        results = {"confusion_matrix":conf_matrix, "accuracy":accuracy, "precision":precision, "recall":recall, "f1":f1}
        return results