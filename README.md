```markdown
# TP3 

Version de l'interpréteur python utilisé : 3.11.8

Avant l'exécution, veuillez utiliser ces commandes dans le terminal :

```
pip install numpy
pip install scikit-learn
```

Pour lancer l'entrainement exécutez le fichier : entrainer_tester.py

Notes :

- Conversion des labels M, F et I en 0, 1, 2 dans le fichier `load_datasets.py`.
- Ajout de la classe `utils.py` au projet pour faciliter la réutilisation du code entre les classes `knn.py` et `naiveBayes.py`.
- Ce travail a été réalisé seul par Jeammy Côté.
- Les réponses aux questions se trouvent dans le fichier `reponses.txt` ainsi qu'en version PDF dans `reponses.pdf`.
- L'exécution du code peut être longue de plusieurs minutes puisque l'algorithme de knn recalcule la distance d'Euclide à chaque exécution de la fonction predict.
- La librairie scikit-learn n'a été utilisé qu'aux questions 2.1.5 et 2.2.4 pour la comparaison avec les résultats des algorithmes codés.
```