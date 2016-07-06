import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import time

# Parametres
n_classes = 3
plot_colors = "byr"
plot_step = 0.02
label_names = ['0', '1', '2']
x_lab = []
y_lab = []

for t in range(1,10):
    debut = time.time()
    # On charge les donnees
    df = pd.read_csv('cwnd.dat', delimiter = ';')
    x = []
    y = []
    dp= pd.read_csv('label.csv')
    for i in range(0,len(df)/t):
        x.append([df['time'].values[i], df['cwnd'].values[i]])
        y.append(dp['label'].values[i])
    y = np.array(y)
    X = np.array(x)

    # Construit un arbre de decision avec les valeurs donnees -> entrainement
    clf = DecisionTreeClassifier().fit(X, y)

    #On choisit les limites en fonction des valeurs dans les ensembles
    x_min, x_max = 10, X[:, 0].max() + 1
    y_min, y_max = 0, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
    #Prediction -> l'algo essaye de predire quel point appartient a quelle classe
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    #contour -> met les labels predits (avec les couleurs) sur le graphe
    fin = time.time()
    temps_tot = fin - debut
    x_lab.append(len(X))
    y_lab.append(temps_tot)

plt.xlabel('Nombre de valeurs')
plt.ylabel('Temps (s)')

plt.scatter(x_lab, y_lab)
plt.suptitle("Performance de l'algorithme : Arbre de decision")
plt.show()
