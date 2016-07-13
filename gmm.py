"""
Auteur : Matthieu Leon

Script permettant d'afficher le resultat  d'une classification
GMM sur les donnees.

"""


import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.cross_validation import StratifiedKFold
from sklearn.externals.six.moves import xrange
from sklearn.mixture import GMM

target_names = ['0', '1', '2']

def make_ellipses(gmm, ax):
    for n, color in enumerate('rgb'):
        v, w = np.linalg.eigh(gmm._get_covars()[n][:2, :2])
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v *= 9
        ell = mpl.patches.Ellipse(gmm.means_[n, :2], v[0], v[1],
                                  180 + angle, color=color)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)

# On charge les donnees d'entrainement et de test
df = pd.read_csv('cwnd.dat', delimiter = ';')
x_train = []
y_train = []
x_test = []
y_test = []
dp= pd.read_csv('label.csv')
for i in range(0,len(df)/3):
    if (i%3 == 0):
        x_train.append([df['time'].values[i], df['cwnd'].values[i]])
        y_train.append(dp['label'].values[i])
    else:
        x_test.append([df['time'].values[i], df['cwnd'].values[i]])
        y_test.append(dp['label'].values[i])

y_test = np.array(y_test)
y_train = np.array(y_train)
X_train = np.array(x_train)
X_test = np.array(x_test)

n_classes = len(np.unique(y_train))

# 4 types de classification GMM avec differentes covariances
classifiers = dict((covar_type, GMM(n_components=n_classes,
                    covariance_type=covar_type, init_params='wc', n_iter=20))
                   for covar_type in ['spherical', 'diag', 'tied', 'full'])

n_classifiers = len(classifiers)

plt.figure(figsize=(3 * n_classifiers / 2, 6))
plt.subplots_adjust(bottom=.01, top=0.95, hspace=.15, wspace=.05,
                    left=.01, right=.99)


for index, (name, classifier) in enumerate(classifiers.items()):

    classifier.means_ = np.array([X_train[y_train == i].mean(axis=0)
                                  for i in xrange(n_classes)])

    # Entrainement non supervise
    classifier.fit(X_train)

    h = plt.subplot(2, n_classifiers / 2, index + 1)
    make_ellipses(classifier, h)

    # Donnees de tests avec des croix
    for n, color in enumerate('rgb'):
        data = X_test[y_test == n]
        plt.plot(data[:, 0], data[:, 1], 'x', color=color)

    #Prediction des classes d'entrainement
    y_train_pred = classifier.predict(X_train)
    train_accuracy = np.mean(y_train_pred.ravel() == y_train.ravel()) * 100
    plt.text(0.05, 0.9, 'Train accuracy: %.1f' % train_accuracy,
             transform=h.transAxes)

    #Prediction des classes de tests (ce qu'on cherche)
    y_test_pred = classifier.predict(X_test)

    # Calcul de la precision : faisable dans ce cas
    test_accuracy = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100
    plt.text(0.05, 0.8, 'Test accuracy: %.1f' % test_accuracy,
             transform=h.transAxes)

    plt.xticks(())
    plt.yticks(())
    plt.xlabel('time')
    plt.ylabel('cwnd')
    plt.title(name)

plt.legend(loc='lower right', prop=dict(size=12))


plt.show()
