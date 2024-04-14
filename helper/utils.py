from math import sqrt

import numpy as np


class Utils:

    def __init__(self, **kwargs):
        """
        C'est un Initializer. 
        Vous pouvez passer d'autre paramètres au besoin,
        c'est à vous d'utiliser vos propres notations
        """
        self.kwargs = kwargs

    def printResults(results):
        for key, value in results.items():
            if key == "confusion_matrix":
                print(f"{key}:")
                print(value)
            else:
                print(f"{key} : {value}")
        print("\n")


    def mean(arrayOfValues):
        return sum(arrayOfValues) / len(arrayOfValues)
    

    def standardDeviation(arrayOfValues):
        mean = Utils.mean(arrayOfValues)
        var = sum([(x - mean) ** 2 for x in arrayOfValues]) / float(len(arrayOfValues) - 1)
        return sqrt(var)
    

    def euclideanDistance(X, Y):
        return sqrt(sum(pow(x - y, 2) for x, y in zip(X, Y)))
    

    def generateConfusionMatrix(test, predictions, test_labels, nbOfClasses):
        """
        Renvoyer la matrice de confusion
        """
        confMatrix = np.zeros((nbOfClasses, nbOfClasses))
        for i in range(len(test)):
            confMatrix[test_labels[i]][predictions[i]] += 1
        return confMatrix
    
    
    ## Obtaining TP, FN, FP from a confusion matrix
    def summariseConfusionMatrix(confMatrix, classes):
        matrixSummarised = np.array([], dtype = int)
        for classe in classes:
            tp = confMatrix[classe][classe]
            fn = confMatrix[classe].sum() - tp
            fp = confMatrix[:,classe].sum() - tp
            classDict = {"tp":tp, "fn":fn, "fp":fp}
            matrixSummarised = np.append(matrixSummarised, classDict)
        return matrixSummarised


    def calculAccuracy(confMatrix, classes):
        matrixSummarised = Utils.summariseConfusionMatrix(confMatrix, classes)
        totalCorrectPredictions = 0
        totalPredictions = confMatrix.sum()
        for classe in matrixSummarised:
            totalCorrectPredictions += classe['tp']
        return totalCorrectPredictions / totalPredictions
    

    ## Calculating Macro-precision with formula : 1/n * sum(TPi / TPi + FPi)
    def calculPrecision(confMatrix, classes):
        matrixSummarised = Utils.summariseConfusionMatrix(confMatrix, classes)
        precisionByClasses = np.array([])
        for classe in classes:
            tp = matrixSummarised[classe]["tp"]
            fp = matrixSummarised[classe]["fp"]
            precision = tp / (tp + fp)
            precisionByClasses = np.append(precisionByClasses, precision)
        precisionMean = precisionByClasses.mean()
        return precisionMean
    

    def calculRecall(confMatrix, classes):
        matrixSummarised = Utils.summariseConfusionMatrix(confMatrix, classes)
        recallByClasses = np.array([])
        for classe in classes:
            tp = matrixSummarised[classe]["tp"]
            fn = matrixSummarised[classe]["fn"]
            recall = tp / (tp + fn)
            recallByClasses = np.append(recallByClasses, recall)
        recallMean = recallByClasses.mean()
        return recallMean  


    def calculF1Score(precision, recall):
        return (2 * precision * recall) / (precision + recall)
