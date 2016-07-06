import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Parametres
n_classes = 3
plot_colors = "bry"
plot_step = 0.02
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

for pairidx, pair in enumerate([0]):

    # Construit un arbre de decision avec les valeurs donnees
    clf = DecisionTreeClassifier().fit(X, y)

    #plt.subplot(2, 3, pairidx + 1)
    #On choisit les limites en fonction des valeurs dans les ensembles
    x_min, x_max = 10, X[:, 0].max() + 1
    y_min, y_max = 0, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
    #Prediction -> l'algo essaye de predire quel point appartient a quelle classe
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    print(len(np.c_[xx.ravel(), yy.ravel()]))

    Z = Z.reshape(xx.shape)
    print(len(Z[1]))
    #contour -> met les labels predits (avec les couleurs) sur le graphe
    cs = plt.contourf(xx, yy, Z)

    #On s'occupe des axes du graphique
    plt.xlabel('time')
    plt.ylabel('cwnd')
    plt.axis("tight")

    # Affichage des points d'entrainement
    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(y == i)#choix label
        plt.scatter(X[idx, 0], X[idx, 1], c=color, label = label_names[i],
                    cmap=plt.cm.Paired)
    plt.axis("tight")

plt.suptitle("Arbre de decision")
plt.legend()
plt.show()
