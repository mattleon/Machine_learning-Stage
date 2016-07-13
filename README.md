===================


Vous trouverez ici les explications sur comment utiliser les différents scripts fournis et quelques rapides explications sur leur fonctionnement.


#### Prérequis

Pour pouvoir utiliser les différents scripts et algorithmes, voici ce qu'il faut sur votre machine :

* Python2.7 ou Python3
* Divers modules python :
	* matplotlib :
	` $ sudo apt-get install python-matplotlib`
	* pandas :
	`$ sudo apt-get install python-pandas`
	* sklearn :
	 `$ sudo apt-get install python-sklearn`
	* numpy :
	`$ sudo apt-get install python-numpy`
* Des fichiers d'entraînement et de test au format csv.


#### Lancer un script

Pour lancer un des script, il suffit de lancer la commande suivante dans un terminal :
> $ python 'nom_du_script'

Pour le moment, le nom des fichiers d'entraînement et/ou de tests ne sont pas à donner en arguments, ils sont déjà présents dans le code, cela pourra être modifié sous peu.

#### Comprendre les différentes algorithmes

Vous avez ici la possibilité d'utiliser différents scripts et, par conséquent, différents algorithmes.
Voici une liste des différents algorithmes : (Cette liste sera mise à jour au fur et à mesure des ajouts et/ou suppressions)

> - Arbre de décision
> - K-plus proches voisins
> - SVC (Support Vector Clustering) (à completer)
> - Soft-voting
> - Classification GMM (Gaussian Mixture Model) (à compléter)

#####  Arbre de décision
  L'Arbre de Décision est une technique d'apprentissage supervisé. Le but de cet algorithme est de créer un modèle prédictif, c'est-à-dire qui va prédire la classe (ou label) d'une variable, ici à deux dimensions. Pour créer ce modèle, l'arbre de construit au fur et à mesure de différentes décisions qu'il jugera bonne ou mauvaise en fonction des valeurs de tests qu'il a. On parle ici d'arbre, car l'algorithme prenant différentes décisions de classification au fur et à mesure de son apprentissage, il va créer des 'branches' qu'il décidera de garder ou non.
  On obtient alors à la fin un arbre prédictif qui, à partir de variables d'entrée, pourra donner sa classe. L'arbre ayant prit différentes décisions et ayant gardées une ou plusieurs branches, il se peut que les résultats fluctuent un peu, mais rien qui ne rende l'algorithme inintéressant d'un point de vue réussite.

##### K-plus proches voisins

	Les K-plus proches voisins est une méthode d'apprentissage supervisé qui raisonne avec le principe suivant : "On définit un point à partir de ceux qui l'entourent".
	A contrario des autres algorithmes, on ne peut pas vraiment utiliser de modèle précis pour l'apprentissage, les voisins différant à chaque jeu de données.
	Le fonctionnement de cet algorithme est donc plutôt simple à comprendre :
		- On choisit un nombre k de voisins à prendre en compte
		- Pour chaque point, l'algorithme va alors regarder les k points les plus proches et, en fonction de ces points, définir la classe du point ciblé.

#### SVC (Support Vector Clustering)

	SVC est une technique d'apprentissage supervisé reposant sur l'algorithme One-vs-One.

##### Soft-Voting

	Soft-voting est une méthode de prédiction se basant sur d'autres algorithmes de prédiction. Pour prédire la classe d'un point, le Soft-Voting considère les prédictions faites par d'autres algorithmes (que nous avons nous-même lancé) et va faire la moyenne des probabilités pour chaque classe.

	Par exemple :

	Supposons que j'utilise 3 algorithmes différents pour trouver la classe d'un point, avec 2 classes différentes possibles.
	Soit les probabilités suivantes : respectivement 0.2, 0.5 et 0.3 pour la classe 1 d'après les 3 algorithmes et 0.8, 0.5 et 0.7 pour la classe 2 d'après les mêmes algorithmes, alors le Soft-Voting fera :

	P(1) = (0.2+0.5+0.3)/3 = 0.33

	P(2) = (0.8 + 0.5 + 0.7)/3 = 0.66

	Le point choisi sera donc de la classe 2 d'après le Soft-Voting.
	De plus, si l'on fait plus confiance à un algorithme qu'à un autre, on peut modifier leur 'poids' pour leur donner plus d'importance.

	Par exemple :

	Reprenons les 3 algorithmes précédents et leurs valeurs. Cette fois, je donne plus d'importante au deuxième algorithme qu'aux deux autres, imaginons que nous lui donnions un poids de 3. On aura alors :

	P(1) = (0.2 + 0.5*3 + 0.3)/(somme_des_poids) = 0.4

	P(2) = (0.8 + 0.5*3 + 0.7)/(somme_des_poids) = 0.6

	Les probabilités ont été légèrement modifiées dans cet exemple.

	Cette modification de l'importance des algorithmes permet de pouvoir donner plus d'importance à un algorithme que l'on pensera donner de meilleurs résultats sans pour autant devoir supprimer un autre que l'on pensera moins intéressant.


#### Enregistrer/charger le résultat d'un entraînement

	- Enregistrement :
	L'enregistrement d'un entraînement se fait grâce au module joblib de sklearn. Il s'utilise simplement en lancant la commande suivante dans le script :
	'joblib.dump(clf, 'model.pkl')'
	Avec clf l'objet correspondant au résultat de la commande fit (l'entraînement) et model.pkl le nom du fichier qui sera enregistré.

	- Chargement :
	Le chargement est tout aussi simple. Il suffit de lancer la commande suivante :
	'clf = joblib.load('model.pkl')'
	avec model.pkl le nom du fichier à charger, et clf qui sera l'entraînement chargé.
	L'objet obtenu peut ensuite être utilisé comme le résultat d'un entraînement normal.
----------
