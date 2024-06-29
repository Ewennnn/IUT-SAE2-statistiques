from operator import itemgetter
import matplotlib.pyplot as plt
import contextily as ctx
import geopandas as gpd
import pandas as pd
import numpy as np
import platform
import random
import os

"""Import des fichiers pour nos propres exemples"""

if platform.system() == 'Windows':
    data0 = pd.read_csv(os.path.dirname(__file__) + '\\donnees_de_test\\data0.csv')
    data1 = pd.read_csv(os.path.dirname(__file__) + '\\donnees_de_test\\data1.csv')
    data2 = pd.read_csv(os.path.dirname(__file__) + '\\donnees_de_test\\data2.csv')
    data3 = pd.read_csv(os.path.dirname(__file__) + '\\donnees_de_test\\data3.csv')
    data4 = pd.read_csv(os.path.dirname(__file__) + '\\donnees_de_test\\data4.csv')

    dataGeo0 = pd.read_csv(os.path.dirname(__file__) + '\\donnes_geo\\csv\\manif.csv')
    dataGeo1 = pd.read_csv(os.path.dirname(__file__) + '\\donnes_geo\\csv\\hebergements_locatifs.csv', sep='\t')
    dataGeo2 = pd.read_csv(os.path.dirname(__file__) + '\\donnes_geo\\csv\\lycees-en-bretagne.csv')
    dataGeo3 = pd.read_csv(os.path.dirname(__file__) + '\\donnes_geo\\csv\\accident-corse.csv')
    dataGeo4 = pd.read_csv(os.path.dirname(__file__) + '\\donnes_geo\\csv\\etablissement-Wifi-Paris.csv')

    contourBzh = os.path.dirname(__file__) + '\\donnes_geo\\fr_geojson\\regions\\bretagne\\departements-bretagne.geojson'
    contourPdl = os.path.dirname(__file__) + '\\donnes_geo\\fr_geojson\\regions\\pays-de-la-loire\\departements-pays-de-la-loire.geojson'
    contourCorse = os.path.dirname(__file__) + '\\donnes_geo\\fr_geojson\\regions\\corse\\departements-corse.geojson'
    contourParis = os.path.dirname(__file__) + '\\donnes_geo\\fr_geojson\\departements\\75-paris\\arrondissements-75-paris.geojson'
else:
    data0 = pd.read_csv(os.path.dirname(__file__) + '/donnees_de_test/data0.csv')
    data1 = pd.read_csv(os.path.dirname(__file__) + '/donnees_de_test/data1.csv')
    data2 = pd.read_csv(os.path.dirname(__file__) + '/donnees_de_test/data2.csv')
    data3 = pd.read_csv(os.path.dirname(__file__) + '/donnees_de_test/data3.csv')
    data4 = pd.read_csv(os.path.dirname(__file__) + '/donnees_de_test/data4.csv')

    dataGeo0 = pd.read_csv(os.path.dirname(__file__) + '/donnes_geo/csv/manif.csv')
    dataGeo1 = pd.read_csv(os.path.dirname(__file__) + '/donnes_geo/csv/hebergements_locatifs.csv', sep='\t')
    dataGeo2 = pd.read_csv(os.path.dirname(__file__) + '/donnes_geo/csv/lycees-en-bretagne.csv')
    dataGeo3 = pd.read_csv(os.path.dirname(__file__) + '/donnes_geo/csv/accident-corse.csv')
    dataGeo4 = pd.read_csv(os.path.dirname(__file__) + '/donnes_geo/csv/etablissement-Wifi-Paris.csv')

    contourBzh = os.path.dirname(__file__) + '/donnes_geo/fr_geojson/regions/bretagne/departements-bretagne.geojson'
    contourPdl = os.path.dirname(__file__) + '/donnes_geo/fr_geojson/regions/pays-de-la-loire/departements-pays-de-la-loire.geojson'
    contourCorse = os.path.dirname(__file__) + '/donnes_geo/fr_geojson/regions/corse/departements-corse.geojson'
    contourParis = os.path.dirname(__file__) + '/donnes_geo/fr_geojson/departements/75-paris/arrondissements-75-paris.geojson'

"""Fonction auxilières aux méthodes de partitionnement"""

def moy(*list):

    """
    Calcul indépendament la moyenne de chaque liste
    """

    ret = [sum(list[i]) / len(list[i]) for i in range(len(list))]
    if len(ret) == 1:
        return ret[0]
    return ret

def moyPartition(partition):

    """
    Retourne le point moyen d'une liste de points
    """

    x = [p[0] for p in partition]
    y = [p[1] for p in partition]
    return [moy(x), moy(y)]


def dist(p1, p2):
    
    """
    Renvoie la distance entre deux points
    """
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def makeRenderFolder(path, folder_to_create : str):
    if not os.path.exists(os.path.join(os.path.dirname(path), folder_to_create)):
        os.makedirs(os.path.join(os.path.dirname(path), folder_to_create))



def Inertie(partitions, centres):
    inerties = [[] for _ in partitions]
    for centre in range(len(centres)):
        inerties[centre] = [dist(centres[centre], p) for p in partitions[centre]]

    return sum([sum(i) for i in inerties])



def voisinage(point, ReferencePoint, MinDist):

    return [p for p in point if dist(p, ReferencePoint) < MinDist]

def extensionClasse(point, ReferecePoint, PointsVoisins : list,C, epsilon, Nmin):

    point[ReferecePoint]["Classe"] = C

    for p in PointsVoisins:
        if not point[p]["marque"]:
            point[p]["marque"] = True
            PtsVoisPrime = voisinage(point, p, epsilon)

            if len(PtsVoisPrime) >= Nmin:
                extensionClasse(point, p, PtsVoisPrime, C, epsilon, Nmin)

            
        if point[p]["Classe"] == -1:
            point[p]["Classe"] = C

"""
Méthode A de partitionnement
"""

def MethodA(data, ListPtsInit, NbMaxIter):

    """
    Effectue un partitionnement selon l'algorithme de la Méthode A
    crée len(ListPtsInit) partitions dans data.

    Calcul la distance des points selon un point moyen donné et les tries en fonction du point le plus proche.

    >>> ListOfCentroids: Liste de points de référence / Liste de points moyen de chaque partition
    >>> ListOfClasses:   Liste contenant tous les points de chaque partition
    
    >>> jdist:          Tableau contenant toutes les distances d'un points à tous les points moyens
    >>> minInListIndex: Variable ayant pour valeur l'index du plus petit élément de jdist
    """

    if NbMaxIter < 1:
        raise ValueError("NbMaxIter doit être suppérieur à 0")
    if len(ListPtsInit) < 1:
        raise Exception("Il est impossible de crée 0 partition")

    ListOfCentroids = ListPtsInit

    for _ in range(NbMaxIter):
        # Réinitialise à chaque itération la liste des partitions
        # pour effectuer une nouvelle partition selon les nouveaux points moyens
        ListOfClasses = [[] for _ in ListPtsInit]

        for j in range(len(data)):
            # Réinitialise à chaque nouveau point les variables
            jdist = []
            point = [data['0'][j], data['1'][j]]

            for k in ListOfCentroids:
                jdist.append(dist(point, k))

            # Détecte la plus petite distance entre les 4 points
            # Ajout de ce point dans la partition du point le plus proche
            minInListIndex = min(enumerate(jdist), key=itemgetter(1))[0]
            ListOfClasses[minInListIndex].append(point)

        # Calcul du point moyen de chaque partition et le met en tant que nouveau point de centre
        for i, part in enumerate(ListOfClasses):
            if len(part) > 0:
                ListOfCentroids[i] = moyPartition(part)

    return ListOfClasses, ListOfCentroids


def MethodAExample(Title: str):

    """
    Crée un graphique du partitionnement de la méthode A
    """

    # Création du partitionnement
    data = data0
    nombre_de_partitions = 4
    nombre_iterations = 50

    randompoints = [[data0['0'][r], data0['1'][r]] for r in [random.randint(0, len(data)) for _ in range(nombre_de_partitions)]]
    pointsList, centroids = MethodA(data, randompoints, nombre_iterations)


    # Création des graphiques
    fig, partitionScatter = plt.subplots()
    plt.title(Title)

    # Affichage des points moyens
    for p in centroids:
        partitionScatter.scatter(p[0], p[1], color='Black', zorder=1)

    # Affichage des points en fonction de leur partition
    for partition in pointsList:
        partitionScatter.scatter([p[0] for p in partition], [p[1] for p in partition], zorder=0)

    partitionScatter.legend(["Points moyens"])

    makeRenderFolder(__file__, "Exemple_PDF_groupe_E.Bosquet_&_L.Barassin")
    fig.savefig(os.path.dirname(__file__) + f"/Exemple_PDF_groupe_E.Bosquet_&_L.Barassin/{Title}.pdf")

"""
Inertie de partitionnement de la méthode A
"""

def MethodAElbow(data, NbClassMax, Titlee : str):

    x = np.arange(1, NbClassMax+1)
    y = []

    # Création des données
    for k in range(1, NbClassMax+1):
        randompoints = [[data['0'][r], data['1'][r]] for r in [random.randint(0, len(data)) for _ in range(k)]]
        classes, centroids = MethodA(data, randompoints, 20)
        
        y.append(Inertie(classes, centroids))

    # Création du graphique
    fig, coude = plt.subplots()

    coude.plot(x, y)
    plt.title(Titlee)

    makeRenderFolder(__file__, "Exemple_PDF_groupe_E.Bosquet_&_L.Barassin")
    fig.savefig(os.path.dirname(__file__) + f"/Exemple_PDF_groupe_E.Bosquet_&_L.Barassin/{Titlee}.pdf")


"""
Méthode A intelligente
"""

def MethodAIntel(data, NbMaxIter):

    # mFixe est notre point moyen qu'on va utiliser du début jusqu'à la fin
    listeDePoints = [[data['0'][p], data['1'][p]] for p in range(len(data))]
    mFixe = moyPartition([[listeDePoints[p][0], listeDePoints[p][1]] for p in range(len(listeDePoints))])

    ListOfCentroids=[]
    while(len(listeDePoints) > NbMaxIter):

        #faut mettre un point de base j'imagine pour remplacer après
        mTemp = [dist(mFixe, [listeDePoints[p][0], listeDePoints[p][1]]) for p in range(len(listeDePoints))] 
        maxPointIndex = max(enumerate(mTemp), key=itemgetter(1))[0]
        mTemp = [listeDePoints[maxPointIndex][0], listeDePoints[maxPointIndex][1]]

        # Ensuite on va chercher le maximum de point qui sont le plus près de notre nouveau point moyen "mTemp" que du point mFixe
        Stemp = []
        for i in range(len(listeDePoints)):
            p = [listeDePoints[i][0], listeDePoints[i][1]]
            if dist(mTemp,p) <= dist(mFixe,p):
                Stemp.append(p)
        
        mTemp = moyPartition(Stemp)

        # Si il y a assez de points qui sont près de mTemp on peut rajouter mTemp dans la liste des points moyens 
        if len(Stemp) > NbMaxIter:
            ListOfCentroids.append(mTemp)

        nouvelleListePoints = []

        # On va enlever tous les points qui sont dans notre Stemp de la listePoints. 
        for i in range(len(listeDePoints)):
            p = [listeDePoints[i][0], listeDePoints[i][1]]
            if p not in Stemp:
                nouvelleListePoints.append(p)
        listeDePoints = nouvelleListePoints

    return MethodA(data, ListOfCentroids, 1)[0], ListOfCentroids

def MethodAIntelExample(Title):

    ListOfClasses, ListOfCentroids = MethodAIntel(data0, 15)

    fig, ax = plt.subplots()

    for p in ListOfCentroids:
        ax.scatter(p[0], p[1], color='Black', zorder=1)

    for partition in ListOfClasses:
        ax.scatter([p[0] for p in partition], [p[1] for p in partition], zorder=0)

    ax.legend(["Points moyens"])
    plt.title(Title)

    makeRenderFolder(__file__, "Exemple_PDF_groupe_E.Bosquet_&_L.Barassin")
    fig.savefig(os.path.dirname(__file__) + f"/Exemple_PDF_groupe_E.Bosquet_&_L.Barassin/{Title}.pdf")

"""
Méthode B de partitionnement
"""

def MethodB(data, epsilon, Nmin):
    if type(data) is list:
        point = list((p[0], p[1]) for p in data)
    else:
        point = list(zip(data['0'], data['1']))

    point = {a: {"marque" : False, "Classe": -1} for a in point}

    C = 0

    for p in point:
        if not point[p]["marque"]:
            point[p]["marque"] = True
            PtsVois = voisinage(point, p, epsilon)

            if len(PtsVois) < Nmin:
                point[p]["Classe"] = -2
            else:
                C += 1
                extensionClasse(point, p, PtsVois, C-1 ,epsilon, Nmin)
    

    return [[point[0], point[1], info["marque"], info["Classe"]] for point, info in point.items()], C

def MethodBExample(Title : str):

    """
    Méthode B de partitionnement
    Renvoie une liste au format demmandé des points avec leur numéro de classe

    Valeurs optimales:
    >>> data0 : 1.5, 3 | 0.7, 3
    >>> data1 : 0.5, 3
    >>> data2 : 0.8, 5
    >>> data3 : Pas meilleur partitionnement
    >>> data4 : 0.07, 5 | 0.08, 5
    """

    fig, ax = plt.subplots()

    points, c = MethodB(data1, 0.5, 3)

    for classe in range(-2, c):
        if classe == -1 or classe == 0:
            pass
        ax.scatter([p[0] for p in points if p[3] == classe], [p[1] for p in points if p[3] == classe], zorder=1)

    ax.legend([f"Bruit {len([p for p in points if p[3] == -2])}"])

    plt.title(Title)
    
    makeRenderFolder(__file__, "Exemple_PDF_groupe_E.Bosquet_&_L.Barassin")
    fig.savefig(os.path.dirname(__file__) + f"/Exemple_PDF_groupe_E.Bosquet_&_L.Barassin/{Title}.pdf")

"""
Comparaison des deux méthodes
Des paramètres par défaut ont été définis (non obligatoire) afin de rendre cette fonction utilisable pour plusieurs données tout en respectant les consignes d'implémentation.
"""
def CompareMethods(Title, data = data0, e = 1.5, N = 3):

    fig, ax = plt.subplots(1, 2)

    partA, centoridsA = MethodAIntel(data, 30)
    partB, nbClassB = MethodB(data, e, N)

    # Affichage méthode A
    for p in centoridsA:
        ax[0].scatter(p[0], p[1], color='Black', zorder=1)

    for partition in partA:
        ax[0].scatter([p[0] for p in partition], [p[1] for p in partition], zorder=0)

    # Affichage méthode B
    for classe in range(-2, nbClassB):
        if classe == -1 or classe == 0:
            pass
        ax[1].scatter([p[0] for p in partB if p[3] == classe], [p[1] for p in partB if p[3] == classe], zorder=1)

    # Titres & légendes
    ax[0].set_title("Méthode A")
    ax[1].set_title("Méthode B")

    ax[0].legend(["Points moyens", f"Classes : {len(partA)}"], loc='upper right')
    ax[1].legend([f"Bruit {len([p for p in partB if p[3] == -2])}", f"Classes : {nbClassB}"], loc='upper right')

    fig.suptitle(Title)

    makeRenderFolder(__file__, "Exemple_PDF_groupe_E.Bosquet_&_L.Barassin")
    fig.savefig(os.path.dirname(__file__) + f"/Exemple_PDF_groupe_E.Bosquet_&_L.Barassin/{Title}.pdf")

"""
Attendu pour la SAE
Nous avons crée une fonction permettant d'afficher n'importe quel région du monde en fonction d'un 'contour' donné.
Ce contour permet de calibrer la carte et d'afficher corretement la région voulue.

Des fonctions nous permettent de trier et convertir les données obtenus dans un CSV depuis des données OpenData pour être exploitées par nos algorithmes.
"""
def verification(donnee):

    """
    Vérifie si une valeur contenue dans le CSV peut être convertie en float
    """

    if type(donnee) is not float:
        for c in str(donnee):
            if c not in "0123456789.-":
                return False

        try:
            float(donnee)
        except:
            return False

        return float(donnee) < 100
    return False

def makePointValues(data):

    """
    Crée un tableau de points ayant pour coordonnées les valeurs de Latitude et de Longitude de nos données CSV
    """

    line_to_delete = []

    for p in range(len(data)):
        if not verification(data["Latitude"][p]) or not verification(data["Longitude"][p]):
            line_to_delete.append(p)

    return [[float(data["Latitude"][p]), float(data["Longitude"][p])] for p in range(len(data)) if p not in line_to_delete]

def creeCarte(donnes : pd.core.frame.DataFrame, contour : str, epsilon, Nmin, title, bruit = False):

    # Lis le fichier et crée le graphique
    data = gpd.read_file(contour)
    fig, axes = plt.subplots(figsize=(10, 10))

    # Definit le CRS de la carte
    data_CRS = data.to_crs(epsg = 4326)

    # Crée les contour des départements
    ax = data.plot(
    ax = axes,
    linewidth = 0.5,
    edgecolor = 'black',
    alpha = 0.5,
    color='white'
    )

    # Crée la carte 
    ctx.add_basemap(ax, crs=data_CRS.crs)

    # Partitionne les données
    points = makePointValues(donnes)
    part, c = MethodB(points, epsilon, Nmin)

    # Affiche sur la carte les résultats
    if bruit:
        plt.scatter([p[1] for p in part if p[3] == -2], [p[0] for p in part if p[3] == -2], s=0.3)

    for classe in range(-1, c):
        plt.scatter([p[1] for p in part if p[3] == classe], [p[0] for p in part if p[3] == classe])

    plt.legend([f"Bruit : {len([p for p in part if p[3] == -2])}", f"Classes : {c}"])
    plt.title(title)
    axes.set_axis_off()

    makeRenderFolder(__file__, "Exemple_PDF_groupe_E.Bosquet_&_L.Barassin")
    fig.savefig(os.path.dirname(__file__) + f"/Exemple_PDF_groupe_E.Bosquet_&_L.Barassin/{title}.pdf")

if __name__ == "__main__":

    MethodAExample("Exemple de partitionnement avec la méthode A")
    MethodAElbow(data0, 10, "Méthode du coude de la méthode A")
    MethodAIntelExample("Méthode A avec sélection intelligente des points moyens")
    MethodBExample("Exemple de partitionnement de la mathode B")

    CompareMethods("Comparaison - Les deux méthodes de partitionnement fonctionnent")
    CompareMethods("Comparaison - La méthode B est plus efficace", data1, e=0.5)
    CompareMethods("Comparaison - La méthode A est plus efficace", data3)

    creeCarte(dataGeo0, contourPdl, 0.1, 60, "Exemple d'affichage de données Pays de Loire")
    creeCarte(dataGeo0, contourPdl, 0.1, 60, "Exemple d'affichage de données Pays de Loire avec bruit", bruit=True)
    creeCarte(dataGeo1, contourPdl, 0.1, 50, "Hebergements locatifs Pays de Loire", bruit=True)
    creeCarte(dataGeo2, contourBzh, 0.08, 4, "Carte des lycées en Bretagne", bruit=True)
    creeCarte(dataGeo3, contourCorse, 0.1, 10, "Accidents de la circulation en Corse (2012-2019)", bruit=True)
    creeCarte(dataGeo4, contourParis, 0.008, 10, "Carte des établissements possédant la Wi-Fi à Paris", bruit=True)