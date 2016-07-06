import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Parametres
n_classes = 3
plot_colors = "byr"
plot_step = 0.02
plot_step = 1
label_names = ['0', '1', '2']

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

# Construit un arbre de decision avec les valeurs donnees -> entrainement
clf = DecisionTreeClassifier().fit(X, y)

#On choisit les limites en fonction des valeurs dans les ensembles
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = 0, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                     np.arange(y_min, y_max, plot_step))
#Prediction -> l'algo essaye de predire quel point appartient a quelle classe
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
"""
Z est un tableau de tableau tq : Z[y][X] = label, avec un ecart
de plot_step
"""

#print(Z[51][1])

#contour -> met les labels predits (avec les couleurs) sur le graphe
cs = plt.contourf(xx, yy, Z)

#On s'occupe des axes du graphique
plt.xlabel('time')
plt.ylabel('cwnd')
plt.axis("tight")

# Affichage des points d'entrainement
for i, color in zip(range(n_classes), plot_colors):
    idx = np.where(y == i)# Quand label == i
    plt.scatter(X[idx, 0], X[idx, 1], c=color, label = label_names[i],
                    cmap=plt.cm.Paired)
plt.axis("tight")

plt.suptitle("Arbre de decision")
plt.legend()
plt.show()
