"""
Auteur : Matthieu Leon

Le script ici-present permet de creer un classifieur de Soft Voting
a l'aide d'un arbre de decision, les k-plus proches voisins et SVC.

Le resultat est un graphique montrant le resultat des quatres algorithmes
utilises.
Il est toutefois facile de 'capturer' les valeurs des predictions pour chaque algorithme,
via la variable Z pour chaque itteration.
"""

from itertools import product
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

plot_step = 0.1

# On charge les donnees
df = pd.read_csv('cwnd.dat', delimiter = ';')
x = []
y = []
dp= pd.read_csv('label.csv')
for i in range(0,len(df)/3):
    x.append([df['time'].values[i], df['cwnd'].values[i]])
    y.append(dp['label'].values[i])
y = np.array(y)
X = np.array(x)

# Training classifiers
clf1 = DecisionTreeClassifier(max_depth=4)
clf2 = KNeighborsClassifier(n_neighbors=7)
clf3 = SVC(kernel='rbf', probability=True)
eclf = VotingClassifier(estimators=[('dt', clf1), ('knn', clf2),
                                    ('svc', clf3)],
                        voting='soft', weights=[3, 1, 1])

# Entrainement des algo
clf1.fit(X, y)
clf2.fit(X, y)
clf3.fit(X, y)
eclf.fit(X, y)

# Choix des minimum/maximum
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = 0, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                     np.arange(y_min, y_max, plot_step))


f, axarr = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(10, 8))

# boucle pour chaque algo
# On fait les predictions pour chaque algo
for idx, clf, tt in zip(product([0, 1], [0, 1]),
                        [clf1, clf2, clf3, eclf],
                        ['Decision Tree (depth=4)', 'KNN (k=7)',
                         'Kernel SVM', 'Soft Voting']):

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # pour chaque sous-graphique, on s'occupe de mettre les couleurs,
    #de n'afficher que les points et afficher les titres.
    axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.4)
    axarr[idx[0], idx[1]].scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    axarr[idx[0], idx[1]].set_title(tt)
plt.show()
