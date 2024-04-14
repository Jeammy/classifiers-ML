import numpy as np
import random

def load_iris_dataset(train_ratio):
    """Cette fonction a pour but de lire le dataset Iris

    Args:
        train_ratio: le ratio des exemples qui vont etre attribués à l'entrainement,
        le reste des exemples va etre utilisé pour les tests.
        Par exemple : si le ratio est 50%, il y aura 50% des exemple (75 exemples) qui vont etre utilisés
        pour l'entrainement, et 50% (75 exemples) pour le test.

    Retours:
        Cette fonction doit retourner 4 matrices de type Numpy: train, train_labels, test, et test_labels
		
        - train : une matrice numpy qui contient les exemples qui vont etre utilisés pour l'entrainement, chaque 
        ligne dans cette matrice représente un exemple d'entrainement.
		
        - train_labels : contient les labels (ou les étiquettes) pour chaque exemple dans train, de telle sorte
          que : train_labels[i] est l'etiquette pour l'exemple train[i]
        
        - test : une matrice numpy qui contient les exemples qui vont etre utilisés pour le test, chaque 
        ligne dans cette matrice représente un exemple de test.
		
        - test_labels : contient les étiquettes pour chaque exemple dans test, de telle sorte
          que : test_labels[i] est l'etiquette pour l'exemple test[i]
    """
    
    random.seed(1) # Pour avoir les meme nombres aléatoires à chaque initialisation.
    
    # Vous pouvez utiliser des valeurs numériques pour les différents types de classes, tel que :
    conversion_labels = {'Iris-setosa': 0, 'Iris-versicolor' : 1, 'Iris-virginica' : 2}
    
    # Le fichier du dataset est dans le dossier datasets en attaché 
    f = open('datasets/bezdekIris.data', 'r')
    
    
    # TODO : le code ici pour lire le dataset
    # REMARQUE très importante : 
	  # remarquez bien comment les exemples sont ordonnés dans 
    # le fichier du dataset, ils sont ordonnés par type de fleur, cela veut dire que 
    # si vous lisez les exemples dans cet ordre et que si par exemple votre ration est de 60%,
    # vous n'allez avoir aucun exemple du type Iris-virginica pour l'entrainement, pensez
    # donc à utiliser la fonction random.shuffle pour melanger les exemples du dataset avant de séparer
    # en train et test.
    lines = f.readlines()
    nbOfData = sum(1 for line in lines)
    train_data = []
    train_labels_data = []
    test_data = []
    test_labels_data = []

    random.shuffle(lines)
    for index, line in enumerate(lines):
        lineData = line.strip().split(',')
        digits = [float(x) for x in lineData[0:-1]]
        label = conversion_labels[lineData[-1]]
        if index < nbOfData * train_ratio:
          train_data.append(digits)
          train_labels_data.append(label)
        else:
          test_data.append(digits)
          test_labels_data.append(label)

    train = np.array(train_data, dtype = float)
    train_labels = np.array(train_labels_data, dtype = int)
    test = np.array(test_data, dtype = float)
    test_labels = np.array(test_labels_data, dtype = int)

    return (train, train_labels, test, test_labels) # Tres important : la fonction doit retourner 4 matrices (ou vecteurs) de type Numpy. 

def load_wine_dataset(train_ratio):
    """Cette fonction a pour but de lire le dataset Binary Wine quality

    Args:
        train_ratio: le ratio des exemples (ou instances) qui vont servir pour l'entrainement,
        le rest des exemples va etre utilisé pour les test.

    Retours:
        Cette fonction doit retourner 4 matrices de type Numpy: train, train_labels, test, et test_labels
		
        - train : une matrice numpy qui contient les exemples qui vont etre utilisés pour l'entrainement, chaque 
        ligne dans cette matrice représente un exemple d'entrainement.
		
        - train_labels : contient les labels (ou les étiquettes) pour chaque exemple dans train, de telle sorte
          que : train_labels[i] est l'etiquette pour l'exemple train[i]
        
        - test : une matrice numpy qui contient les exemples qui vont etre utilisés pour le test, chaque 
        ligne dans cette matrice représente un exemple de test.
		
        - test_labels : contient les étiquettes pour chaque exemple dans test, de telle sorte
          que : test_labels[i] est l'etiquette pour l'exemple test[i]
    """
    
    random.seed(1) # Pour avoir les meme nombres aléatoires à chaque initialisation.

    # Le fichier du dataset est dans le dossier datasets en attaché 
    f = open('datasets/binary-winequality-white.csv', 'r')

	
    # TODO : le code ici pour lire le dataset
    lines = f.readlines()
    nbOfData = sum(1 for line in lines)
    train_data = []
    train_labels_data = []
    test_data = []
    test_labels_data = []

    random.shuffle(lines)
    for index, line in enumerate(lines):
        lineData = line.strip().split(',')
        digits = [float(x) for x in lineData[0:-1]]
        label = lineData[-1]
        if index < nbOfData * train_ratio:
          train_data.append(digits)
          train_labels_data.append(label)
        else:
          test_data.append(digits)
          test_labels_data.append(label)

    train = np.array(train_data, dtype = float)
    train_labels = np.array(train_labels_data, dtype = int)
    test = np.array(test_data, dtype = float)
    test_labels = np.array(test_labels_data, dtype = int)

    return (train, train_labels, test, test_labels) # La fonction doit retourner 4 structures de données de type Numpy.

def load_abalone_dataset(train_ratio):
    """
    Cette fonction a pour but de lire le dataset Abalone-intervalles

    Args:
        train_ratio: le ratio des exemples (ou instances) qui vont servir pour l'entrainement,
        le rest des exemples va etre utilisé pour les test.

    Retours:
        Cette fonction doit retourner 4 matrices de type Numpy: train, train_labels, test, et test_labels

        - train : une matrice numpy qui contient les exemples qui vont etre utilisés pour l'entrainement, chaque 
        ligne dans cette matrice représente un exemple d'entrainement.

        - train_labels : contient les labels (ou les étiquettes) pour chaque exemple dans train, de telle sorte
          que : train_labels[i] est l'etiquette pour l'exemple train[i]

        - test : une matrice numpy qui contient les exemples qui vont etre utilisés pour le test, chaque 
        ligne dans cette matrice représente un exemple de test.

        - test_labels : contient les étiquettes pour chaque exemple dans test, de telle sorte
          que : test_labels[i] est l'etiquette pour l'exemple test[i]
    """
    conversion_labels = {'M': 0, 'F' : 1, 'I' : 2}
    f = open('datasets/abalone-intervalles.csv', 'r') # La fonction doit retourner 4 matrices (ou vecteurs) de type Numpy.
    lines = f.readlines()
    nbOfData = sum(1 for line in lines)
    train_data = []
    train_labels_data = []
    test_data = []
    test_labels_data = []

    random.seed(1) # Pour avoir les meme nombres aléatoires à chaque initialisation.
    random.shuffle(lines)
    for index, line in enumerate(lines):
        lineData = line.strip().split(',')
        digits = [float(x) for x in lineData[1:]]
        label = conversion_labels[lineData[0]]
        if index < nbOfData * train_ratio:
          train_data.append(digits)
          train_labels_data.append(label)
        else:
          test_data.append(digits)
          test_labels_data.append(label)

    train = np.array(train_data, dtype = float)
    train_labels = np.array(train_labels_data, dtype = int)
    test = np.array(test_data, dtype = float)
    test_labels = np.array(test_labels_data, dtype = int)

    return (train, train_labels, test, test_labels)