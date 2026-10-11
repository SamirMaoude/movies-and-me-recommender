# Movies & Me Recommender

Backend/ML séparé de l'application Movies-And-Me. Le projet utilise
le modèle ALS fourni par `implicit` pour se concentrer sur le MLOps :
entraînement reproductible, évaluation, déploiement et suivi en production.
L'API reçoit les favoris et renvoie des recommandations avec des IDs TMDB.

La [feuille de route](ROADMAP.md) décrit cette direction. Les phases 1 à 8
sont réalisées : ALS est entraîné, comparé aux références et rechargeable
à partir de ses fichiers ou d'une version du registre MLflow. Chaque essai
conserve les informations sur le code, les données et l'environnement.
Les données et les sorties du pipeline sont versionnées avec DVC et leur
récupération depuis S3 est vérifiée. L'API FastAPI sert une version précise
du modèle. La suite porte sur Docker et l'intégration continue.
Le modèle par genres reste disponible pour les comparaisons.

## Phase 1 : exploration de MovieLens

La phase 1 est réalisée dans [notebooks/01_exploration.ipynb](notebooks/01_exploration.ipynb).
Le notebook est exécuté et contient les tableaux, quatre graphiques, les
contrôles de qualité et le résumé final. Le chargement et les calculs
réutilisables sont dans [src/data/exploration.py](src/data/exploration.py), avec
des tests sur des données synthétiques dans [tests/test_exploration.py](tests/test_exploration.py).

Les CSV fournis sont conservés à leur emplacement actuel, `data/raw/` :
`movies.csv`, `ratings.csv`, `links.csv`, `tags.csv`. L'analyse ne les modifie
pas et n'effectue aucun appel TMDB. Source : MovieLens `ml-latest-small`,
snapshot généré le 26 septembre 2018. La description, la licence et la citation
du dataset sont conservées dans [data/raw/README.md](data/raw/README.md).

### Résultats mesurés

| Indicateur | Résultat |
|---|---:|
| Films du catalogue | 9 742 |
| Utilisateurs ayant des notes | 610 |
| Notes | 100 836 |
| Notes par utilisateur, moyenne / médiane | 165,305 / 70,5 |
| Notes par film du catalogue, moyenne / médiane | 10,351 / 3 |
| Note globale moyenne | 3,501557 / 5 |
| Densité de la matrice utilisateurs × films | 1,696827 % |
| Films avec un ID TMDB renseigné | 9 734 / 9 742 (99,917881 %) |
| Labels de genres réels | 19 |
| Applications de tags | 3 683 |

La couverture est calculée après une jointure gauche depuis tous les films :

```python
movies_with_links = movies.merge(links, on="movieId", how="left", validate="one_to_one")
tmdb_coverage = movies_with_links["tmdbId"].notna().mean()
```

Elle mesure la présence d'un ID dans le snapshot, sans vérifier sa disponibilité
actuelle dans TMDB. Le dictionnaire final conserve cette couverture sous forme
de fraction entre 0 et 1. Les moyennes de nombres de notes par film incluent
les films non notés ; les moyennes de scores les excluent naturellement.

### Observations et problèmes de données

- Aucun doublon de ligne ou de clé film, aucune paire utilisateur/film dupliquée
  dans les notes et aucune référence orpheline. Les scores respectent l'échelle
  0,5–5,0 par pas de 0,5.
- Les seuls champs CSV manquants sont **8 `tmdbId`**, concernant 13 notes.
  L'ID TMDB **4912** correspond aux `movieId` **6003 et 144606**, deux entrées
  de *Confessions of a Dangerous Mind (2002)* aux genres différents. Cette
  ambiguïté doit être résolue avant une conversion inverse TMDB → MovieLens.
- **18 films n'ont aucune note** et **66,45 % du catalogue a au plus cinq notes**.
  L'activité est très déséquilibrée et les moyennes de films peu notés sont fragiles.
- **34 films** portent `(no genres listed)` : ce marqueur est exclu des genres
  réels. Le vocabulaire observé contient notamment `Children` et `IMAX` ; ce
  dernier décrit un format de projection et mérite un choix explicite pour la baseline.
- **5 titres sont partagés par plusieurs IDs** ; 13 titres n'ont pas d'année
  finale extractible. Utiliser `movieId` comme clé plutôt que les titres.
- Les tags couvrent **1 572 films (16,14 %)** et proviennent de 58 utilisateurs.
  Un contributeur représente **40,92 %** des tags. Plusieurs tags sur une même
  paire utilisateur/film sont légitimes.
- Les notes vont du **29 mars 1996 au 24 septembre 2018 (UTC)** : ce snapshot
  permet d'expérimenter, mais ne représente pas le catalogue actuel.

MovieLens est exploitable pour une première expérimentation, les genres
permettent une baseline simple, et presque tous les films peuvent être reliés
à un ID TMDB, sous réserve des absences et ambiguïtés recensées. Cette analyse
ne mesure aucune performance de recommandation.

Cette exploration sert de base à la première baseline par genres décrite
ci-dessous.

## Phase 2 : baseline content-based par genres

[notebooks/02_content_based_baseline.ipynb](notebooks/02_content_based_baseline.ipynb)
contient l'expérimentation exécutée, l'inspection des features, un graphique
du profil utilisateur et deux exemples de recommandations. Le prototype validé
a été extrait dans [src/models/content_based.py](src/models/content_based.py) ;
le notebook importe ce module. Les tests synthétiques sont dans
[tests/test_content_based.py](tests/test_content_based.py).

Les features sont les **genres uniquement** : un genre reste un token entier,
y compris `Sci-Fi` et `Film-Noir`. Les valeurs vides et `(no genres listed)`
sont ignorées. `IMAX`, un format de projection, est exclu par défaut ;
`include_imax=True` permet d'inspecter la variante qui le conserve. Aucun gain
prédictif n'est attribué à ce choix sans évaluation.

Le modèle construit une matrice TF-IDF sparse de **9 742 films × 18 features**,
avec **34 vecteurs nuls**. Il calcule le profil moyen des favoris distincts,
puis la similarité cosinus avec les films du catalogue. Il exclut les favoris
et retourne jusqu'à `n` films de score strictement positif. Les notes et les
tags n'interviennent pas dans le classement ; aucune matrice de similarités
film × film n'est construite.

```python
from src.data.exploration import load_movielens
from src.models.content_based import ContentBasedRecommender

movies = load_movielens("data/raw")["movies"]
recommender = ContentBasedRecommender(movies)

# movieId MovieLens : Star Wars IV, Star Wars V, Matrix.
favorites = [260, 1196, 2571]
recommendations = recommender.recommend(favorites, n=10)
print(recommendations)  # Colonnes : movieId, title, genres, score.
```

**Les entrées et sorties utilisent les `movieId` MovieLens, pas les IDs TMDB.**
Les deux profils du notebook (science-fiction et animation) produisent chacun
10 films distincts hors favoris, avec des scores dans [0, 1]. Les premiers
résultats du profil science-fiction incluent *Waterworld*, *Stargate* et
*Demolition Man* ; ceux du profil animation incluent *An American Tail*,
*A Bug's Life* et *Jimmy Neutron: Boy Genius*.

Les favoris répétés sont dédupliqués. Les listes vides, IDs inconnus, profils
sans genre utilisable et valeurs de `n` non entières ou non positives lèvent
un `ValueError` explicite. Si moins de `n` candidats partagent des genres,
la liste est plus courte ; si aucun candidat ne reste, le tableau est vide.
Un favori sans genre peut accompagner des favoris exploitables sans ajouter
de signal au profil.

Les scores mesurent une **ressemblance entre genres**, pas une probabilité
d'apprécier un film ni une note prédite. Les ex æquo sont fréquents ; ils sont
départagés par `movieId` croissant pour rendre les sorties reproductibles.
Ce choix privilégie arbitrairement les petits IDs et n'est pas un indicateur
de qualité. La phase 4 présente l'évaluation de ce classement.

Cette version reste disponible pour les essais avec les IDs MovieLens.
La version utilisant les IDs de l'application est décrite ci-dessous.

## Phase 3 : IDs TMDB

[src/data/id_mapper.py](src/data/id_mapper.py) lit les correspondances de
`links.csv`. Il permet de retrouver un ID TMDB à partir d'un film MovieLens,
ou de faire le chemin inverse. Un ID absent renvoie `None`.

```python
from src.data.exploration import load_movielens
from src.data.id_mapper import IdMapper
from src.models.content_based import ContentBasedRecommender
from src.models.tmdb_recommender import TMDBRecommender

tables = load_movielens("data/raw")
mapper = IdMapper(tables["links"])

print(mapper.to_tmdb_id(2571))  # 603 : Matrix
print(mapper.to_movie_id(603))  # 2571

recommender = TMDBRecommender(ContentBasedRecommender(tables["movies"]), mapper)
recommendations = recommender.recommend(
    favorite_tmdb_ids=[27205, 157336, 155, 603],
    n=10,
)
print(recommendations)
# [{"tmdb_id": 5550, "score": 0.9497038041238557}, ...]
```

[src/models/tmdb_recommender.py](src/models/tmdb_recommender.py) utilise le
même calcul par genres, avec des IDs TMDB en entrée et en sortie. Le résultat
est une liste de dictionnaires contenant `tmdb_id` et `score`. Chaque film
apparaît une seule fois, et les favoris sont exclus.

Les IDs TMDB inconnus, ou liés à des films absents du modèle, sont ignorés.
S'il ne reste aucun favori connu, la liste est vide. Elle est aussi vide si
les favoris n'ont aucun genre utilisable. Répéter un favori ne change pas les
scores. Les IDs et `n` doivent être des nombres entiers positifs ; les valeurs
incorrectes provoquent une `ValueError`.

Les films sans ID TMDB et les doublons sont retirés avant de limiter les
résultats à `n`. Un film sans lien ne prend donc pas la place d'un résultat
que l'application pourrait afficher.

L'ID TMDB **4912** correspond aux MovieLens **6003 et 144606**. Le mapping
choisit le plus petit ID MovieLens ; pour les recommandations, il choisit le
plus petit présent dans le modèle. Les deux correspondances restent visibles
avec `mapper.movie_ids_for_tmdb(4912)` et `mapper.ambiguous_mappings`. Si cet
ID TMDB est un favori, aucune de ses deux entrées ne revient dans les résultats.
Ce choix est une règle locale : les fichiers fournis ne permettent pas de
confirmer laquelle des deux entrées décrit correctement le film.

La section TMDB du notebook de baseline montre cet exemple et les cas d'IDs
absents. Les tests sont dans [tests/test_id_mapper.py](tests/test_id_mapper.py)
et [tests/test_tmdb_recommender.py](tests/test_tmdb_recommender.py). Aucun appel
à l'API TMDB et aucune nouvelle dépendance ne sont nécessaires.

## Phase 4 : évaluer les modèles

[src/evaluation/data.py](src/evaluation/data.py) prépare les interactions
et les sépare dans le temps. Une note **supérieure ou égale à 4/5** compte
comme un avis positif. Ces avis servent de favoris simulés pour MovieLens.
Le seuil est configurable. Les notes plus faibles sont conservées pour l'audit
des données ; seuls les favoris transmis sont exclus des recommandations,
comme dans l'application qui ne fournit aucune note ni historique de visionnage.

Les films sont reliés à TMDB avec la règle de la phase 3. Pour une même paire
utilisateur/film TMDB, on garde la première interaction, puis le plus petit
ID MovieLens d'origine en cas d'égalité de date. Une meilleure note ultérieure
ne remplace pas la première. Ce choix évite de placer le même film dans
l'historique et dans les réponses attendues sous deux IDs différents ; il
ne cherche pas à représenter les changements d'avis.

Les dates sont fixées dans [config/evaluation.json](config/evaluation.json).
Elles correspondent aux quantiles temporels proches de 80 % et 90 %, calculés
une fois sur les interactions préparées, avec la méthode `nearest`. Leur
choix utilise les dates et les volumes, sans mesurer la qualité d'un modèle.
La commande ne recalcule pas ces coupures lorsqu'on la relance.

| Groupe | Période UTC | Interactions | Avis positifs |
|---|---|---:|---:|
| Apprentissage | Avant le 22 mars 2016 à 08:26:11 | 80 657 | 39 251 |
| Validation | À partir de cette date, avant le 21 juin 2017 à 07:17:25 | 10 082 | 5 197 |
| Test | À partir du 21 juin 2017 à 07:17:25 | 10 083 | 4 127 |

Ces dates sont communes à tous les utilisateurs. Une interaction exactement
à la date de coupure appartient au groupe suivant. Aucune sélection des
utilisateurs selon leur activité future n'est appliquée.

Depuis la racine du dépôt, produire le
[rapport de préparation](reports/evaluation_data.json) avec :

```powershell
.\.venv\Scripts\python.exe -m src.evaluation.prepare
```

La commande lit les CSV bruts et écrit seulement le rapport JSON. Les fonctions
de préparation renvoient les tables pour la suite. Le rapport conserve le
seuil, les dates, les empreintes des sources et les comptes des lignes retirées :
**13 notes sans ID TMDB** et **1 interaction ultérieure sur le même film**.
Il reste **100 822 interactions**, dont **48 575 avis positifs**. Les CSV
bruts et les notebooks ne sont pas modifiés.

### Limite du découpage

Le catalogue de comparaison comprend les **5 223 films** avec au moins
un avis positif dans l'apprentissage. Dans la validation, **4 595 avis
positifs sur 5 197 (88,4 %)** concernent ces films ; les **602 autres** sont
hors de ce catalogue.

La présence d'un film dans le catalogue ne garantit pas que son utilisateur
dispose déjà de favoris connus. Seuls **20 des 62 utilisateurs** de validation
ont à la fois un historique positif dans l'apprentissage et au moins une
cible positive dans ce catalogue. Cela représente **333 cibles**, soit
**6,4 %** des avis positifs de validation. La comparaison sur ce groupe
reste donc limitée à un petit échantillon. Les autres cas restent présents
dans les tables et comptés dans le rapport.

Les tests de préparation et de couverture sont dans
[tests/test_evaluation_data.py](tests/test_evaluation_data.py) et
[tests/test_evaluation_prepare.py](tests/test_evaluation_prepare.py).
Le test final est décrit par ses volumes uniquement ; il ne sert pas à
entraîner, choisir ou évaluer ces deux références.

### Classement par popularité

[src/models/popularity.py](src/models/popularity.py) compte, pour chaque film
TMDB, les utilisateurs distincts ayant un avis positif dans les données
**d'apprentissage uniquement**. Un même utilisateur compte une seule fois
pour un film, même si la paire apparaît plusieurs fois. Les autres notes
n'ajoutent aucun poids au classement.

Les films sont classés par nombre d'utilisateurs décroissant, puis par ID TMDB
croissant en cas d'égalité. Le score est ce nombre d'utilisateurs, renvoyé
comme un nombre décimal ; il peut dépasser 1 et ne représente pas une probabilité.
Les favoris sont retirés avant de limiter les résultats. Une liste vide ou
des favoris inconnus renvoient le classement global ; un apprentissage sans
avis positif renvoie une liste vide.

Exemple depuis la racine du dépôt :

```python
import json
from pathlib import Path

from src.data.exploration import load_movielens
from src.evaluation.data import prepare_interactions, temporal_split
from src.models.popularity import PopularityRecommender

config = json.loads(Path("config/evaluation.json").read_text(encoding="utf-8"))
tables = load_movielens("data/raw")
interactions, _ = prepare_interactions(
    tables["ratings"], tables["links"], config["positive_threshold"],
)
splits = temporal_split(interactions, config["train_end"], config["validation_end"])
recommender = PopularityRecommender(splits["train"])
recommendations = recommender.recommend([27205, 157336, 155, 603], n=10)
print(recommendations)
```

Ce scénario renvoie les IDs TMDB
`[278, 680, 13, 274, 11, 197, 424, 629, 1891, 550]`.
Les trois premiers sont *The Shawshank Redemption* (218 utilisateurs positifs),
*Pulp Fiction* (208) et *Forrest Gump* (203). Matrix, ID TMDB 603, est retiré
car il fait partie des favoris transmis. Ces comptes proviennent du groupe
d'apprentissage ; ils ne mesurent pas la qualité des recommandations.

Les tests du classement et des cas limites sont dans
[tests/test_popularity.py](tests/test_popularity.py). Aucune nouvelle dépendance
n'est nécessaire. Le modèle par genres et son adaptateur TMDB restent disponibles.

### Comparaison sur la validation

Depuis la racine du dépôt :

```powershell
.\.venv\Scripts\python.exe -m src.evaluation.compare
```

La commande produit [reports/baseline_comparison.json](reports/baseline_comparison.json).
Le [compte rendu](reports/baseline_comparison.md) présente les résultats et
leurs limites. Pour les reproduire, les fichiers bruts et la configuration
doivent être les mêmes ; leurs empreintes et les versions des bibliothèques
sont enregistrées dans le rapport JSON.

Les deux modèles reçoivent les mêmes favoris d'apprentissage et peuvent
recommander les mêmes 5 223 films. Le TF-IDF des genres est ajusté sur les
métadonnées canoniques de ces films uniquement. Les notes de validation ne
construisent ni les profils, ni la popularité, ni le vocabulaire des genres.
Les notes faibles ne retirent aucun candidat. Le modèle par genres garde
ses règles existantes, sans ajout de popularité pour compléter ses résultats.

Les métriques principales utilisent les cibles de validation présentes dans
le catalogue. Recall@10 mesure la part retrouvée dans les dix résultats.
NDCG@10 tient aussi compte de leur position : un film pertinent en tête
rapporte davantage. Chaque utilisateur a le même poids dans la moyenne,
quel que soit son nombre de cibles. Un résultat vide compte pour zéro ;
une cohorte vide donne des métriques `null`.

| Modèle | Recall@10 moyen | NDCG@10 moyen |
|---|---:|---:|
| Popularité | 4,57 % | 0,03894 |
| Genres | 0 % | 0 |

La popularité retrouve cinq cibles chez quatre des vingt utilisateurs.
Les genres ne retrouvent aucune cible dans leurs dix premiers résultats.
Ces mesures concernent **20 utilisateurs et 333 cibles dans le catalogue**,
sur 62 utilisateurs et 5 197 avis positifs de validation. Elles ne permettent
pas de conclure sur tous les utilisateurs de l'application.

Pour ces mêmes vingt utilisateurs, le rapport calcule aussi les scores en
comptant leurs **77 cibles hors catalogue** dans les réponses attendues.
Le Recall@10 de la popularité devient **4,16 %** et son NDCG@10 **0,03481**.
Les 602 cibles hors catalogue de l'ensemble de la validation restent comptées
dans la couverture. Aucun utilisateur sans favoris n'entre dans cette
comparaison ; le comportement de secours reste à évaluer séparément.

Les calculs et le protocole sont vérifiés dans
[tests/test_evaluation_metrics.py](tests/test_evaluation_metrics.py) et
[tests/test_evaluation_compare.py](tests/test_evaluation_compare.py).
La phase 5 utilise cette comparaison comme référence pour ALS.

## Phase 5 : ALS avec implicit

[src/models/collaborative.py](src/models/collaborative.py) utilise ALS sur CPU
avec `implicit==0.7.3`. Le modèle apprend à partir des favoris simulés de
MovieLens. La matrice contient **521 utilisateurs MovieLens × 5 223 films**
et **39 251 paires positives**. Chaque paire vaut 1 : une note de 5/5 ne
pèse pas davantage qu'une note de 4/5. `implicit` applique le coefficient
`alpha` une seule fois à l'entraînement et une seule fois au calcul du profil
d'une requête.

### Entraîner et comparer

Depuis la racine du dépôt, avec l'environnement Python 3.12 créé :

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.training.train
```

La commande [src/training/train.py](src/training/train.py) lit les coupures
de [config/evaluation.json](config/evaluation.json) et les trois essais de
[config/als.json](config/als.json). Elle entraîne chacun sur l'apprentissage,
puis compare ALS, popularité et genres avec le protocole de phase 4.
Le choix donne la priorité au **NDCG@10**, puis au Recall@10 en cas d'égalité ;
une égalité complète conserve le premier essai déclaré.

| Modèle ou essai | Recall@10 moyen | NDCG@10 moyen |
|---|---:|---:|
| Popularité | 4,57 % | 0,03894 |
| Genres | 0 % | 0 |
| ALS, 32 facteurs, régularisation 0,1 | 5,69 % | 0,04708 |
| **ALS, 64 facteurs, régularisation 0,1 — retenu** | **4,81 %** | **0,06231** |
| ALS, 32 facteurs, régularisation 1 | 5,36 % | 0,04429 |

Les trois essais partagent `alpha=20`, `iterations=30`, `random_state=42`
et `num_threads=1`. Le modèle à 64 facteurs obtient le meilleur NDCG ; celui
à 32 facteurs a le meilleur Recall. Le choix ne maximise donc pas les deux
métriques à la fois. Le modèle retenu retrouve sept cibles chez cinq des
vingt utilisateurs.

Les mesures principales portent toujours sur **20 utilisateurs et 333 cibles
connues**. Avec leurs 77 cibles hors catalogue, le Recall d'ALS devient
**3,82 %**, contre **4,16 %** pour la popularité. Son NDCG reste plus élevé :
**0,05612**, contre **0,03481**. Le gain dépend donc de la métrique et des
cibles comptées. Les 333 cibles connues représentent 6,4 % des 5 197 avis positifs
de validation ; le secours pour les profils sans favori connu n'y est pas mesuré.

Le [compte rendu ALS](reports/als_comparison.md) détaille ces résultats.
Le [rapport JSON](reports/als_comparison.json) conserve les scores exacts,
les classements par utilisateur, les configurations, les versions des
bibliothèques et les empreintes des sources. Le
[notebook ALS exécuté](notebooks/03_implicit_als.ipynb) montre les essais,
deux profils de favoris et la sauvegarde/relecture du modèle.

### Charger le modèle et recommander

La commande sauvegarde l'essai ALS retenu dans `artifacts/implicit-als/`.
Ce dossier contient les facteurs, la matrice d'apprentissage, les IDs, le
classement de popularité et un manifeste avec configuration, métadonnées
et empreintes. Le chargement contrôle leur cohérence et ne lit aucun CSV.
`artifacts/` est ignoré par Git. Le pipeline DVC décrit plus bas conserve
sa propre sortie de modèle ; les fichiers d'un run MLflow peuvent aussi être
rechargés avec le module de registre.

```python
from src.models.collaborative import ImplicitRecommender

recommender = ImplicitRecommender.load("artifacts/implicit-als")
favorites = [27205, 157336, 155, 603]
result = recommender.recommend_with_method(favorites, n=10)
print(result["method"])  # als
print(result["recommendations"])
# [{"tmdb_id": 120, "score": ...}, ...]

print(recommender.recommend_with_method([], n=3))
# method = popularity ; IDs 278, 680, 13.
```

Chaque appel construit le profil avec les favoris TMDB connus et
`recalculate_user=True`. Aucun ID utilisateur MovieLens n'est demandé et
les appels ne modifient pas les facteurs partagés. Les favoris répétés ne
comptent qu'une fois. Les favoris inconnus sont ignorés ; s'il ne reste
aucun favori connu, le modèle utilise la popularité de l'apprentissage.
La méthode employée est indiquée par `method` (`als` ou `popularity`).
Seuls les favoris transmis sont retirés des résultats. Les scores ALS
servent au classement et peuvent sortir de l'intervalle [0, 1].

L'adaptateur [TMDBRecommender](src/models/tmdb_recommender.py) reste celui
du modèle par genres. `ImplicitRecommender` utilise directement les IDs
TMDB en entrée et en sortie.

Un dossier de modèle existant est conservé : la commande refuse de l'écraser.
Pour refaire un essai, choisir un nouveau dossier et un rapport distinct :

```powershell
.\.venv\Scripts\python.exe -m src.training.train --artifact-dir artifacts/implicit-als-run-2 --output reports/als_comparison-run-2.json
```

Le modèle sauvegardé reste entraîné sur l'apprentissage uniquement pour
reproduire les scores de validation. Le test final n'a pas été utilisé.
Le réentraînement sur apprentissage et validation, suivi de la mesure finale
sur le test, reste à faire une fois les choix figés.

Les tests sont dans [tests/test_collaborative.py](tests/test_collaborative.py)
et [tests/test_training.py](tests/test_training.py). Le suivi des essais est
décrit ci-dessous.

## Phase 6 : suivi et traçabilité avec MLflow

Chaque configuration ALS produit un run dans l'expérience
`movies-and-me-recommender`. Il contient les paramètres, les métriques de
validation, la durée d'entraînement, l'évaluation détaillée et les quatre
fichiers nécessaires pour recharger ce modèle.

Depuis la racine du dépôt, dans **cmd**, avec `.venv` activé :

```bat
set "MLFLOW_TRACKING_URI=sqlite:///mlruns.db"
python -m src.training.train --artifact-dir artifacts/implicit-als-mlflow-03 --output reports/als_comparison-mlflow-03.json
```

La variable vaut pour ce terminal et les commandes qu'il lance. La base
`mlruns.db` est créée dans le dossier courant. Le dossier du modèle doit
être nouveau : changer le suffixe `03` s'il existe déjà. Les fichiers locaux
de MLflow et `artifacts/` sont ignorés par Git.

Pour consulter les résultats, ouvrir un autre terminal dans le même dossier,
activer `.venv`, puis lancer :

```bat
python -m mlflow server --backend-store-uri sqlite:///mlruns.db --host 127.0.0.1 --port 5000 --workers 1
```

L'interface est accessible à <http://127.0.0.1:5000>. `--workers 1` évite le
problème de lancement de plusieurs processus rencontré sous Windows.

Les fichiers de chaque run se présentent ainsi :

```text
evaluation.json
provenance.json
model/
    model.npz
    interactions.npz
    catalog.json
    manifest.json
```

Le run de l'essai ALS sélectionné contient aussi `comparison.json`, le rapport
final avec les trois configurations et les deux références. Chaque essai du
rapport a son `run_id` ; `selection.selected_run_id` identifie le run retenu.
Le manifeste du modèle conserve également son `run_id` et sa provenance.

`provenance.json` enregistre le commit Git, la présence de modifications
locales (`dirty`), les empreintes SHA-256 de `ratings.csv`, `links.csv` et
`movies.csv`, ainsi que les versions de Python et des dépendances, dont
MLflow. Les tags `git.commit` et `git.dirty` rendent les informations Git
visibles dans l'interface. Les dates de séparation et le seuil positif sont
enregistrés parmi les paramètres et dans les métadonnées du modèle.

Si `dirty` vaut `true`, le commit seul ne permet pas de retrouver tout le code
exécuté : les modifications locales ne sont pas archivées par ce suivi.
Avec les mêmes données, paramètres et environnement, les scores et les
classements peuvent être comparés entre exécutions ; les identifiants des runs
et les durées d'entraînement varient. Le modèle reste entraîné sur
l'apprentissage uniquement, et le test final reste à l'écart.

### Enregistrer une version du modèle

[src/models/registry.py](src/models/registry.py) utilise les fichiers du modèle
déjà présents dans un run : aucun nouvel entraînement n'est lancé.
Dans les commandes suivantes, remplacer `RUN_ID_DU_RAPPORT` par la valeur de
`selection.selected_run_id` du rapport produit plus haut. Garder la même
variable `MLFLOW_TRACKING_URI` dans le terminal.

```bat
python -m src.models.registry register --run-id RUN_ID_DU_RAPPORT --name movies-and-me-als --alias champion
```

La commande affiche un JSON avec `name`, `version`, `run_id`, `model_uri` et
`alias`. L'URI retournée désigne une version exacte, par exemple
`models:/movies-and-me-als/1`. Relancer l'enregistrement du même run sous le
même nom réutilise cette version.

Sans `--alias champion`, la commande enregistre seulement une version
candidate. L'option déplace explicitement l'alias après avoir rechargé la
version et comparé ses recommandations à celles du modèle source. Les IDs
doivent correspondre et les scores sont comparés avec `rtol=1e-6`,
`atol=1e-7`. Si la vérification échoue, l'alias existant reste en place.

Une version sert de référence fixe ; l'alias `champion` peut désigner une
autre version plus tard. Un service doit donc conserver le numéro exact
chargé. Cette vérification contrôle le rechargement du modèle ; la décision
de qualité et la promotion automatique appartiennent à la phase 12.

### Charger depuis le registre et recommander

Dans le même terminal :

```bat
python -m src.models.registry recommend --model-uri models:/movies-and-me-als@champion --favorites 27205 157336 155 603 --n 10
```

Le résultat JSON contient le nom, la version exacte, le `run_id`, l'URI du
modèle, la méthode (`als` ou `popularity`) et les recommandations TMDB.

Le même chargement est disponible en Python :

```python
from src.models.registry import load_recommender

loaded = load_recommender("models:/movies-and-me-als@champion")
print(loaded.name, loaded.version, loaded.run_id, loaded.model_uri)
result = loaded.recommend_with_method([27205, 157336, 155, 603], n=10)
print(result)
```

`load_recommender(...)` accepte aussi un dossier local, `runs:/RUN_ID/model`
ou `models:/movies-and-me-als/1` pour choisir une version précise. L'alias est
résolu une seule fois au chargement ; les recommandations suivantes utilisent
cette même version. `loaded.model` donne accès à l'`ImplicitRecommender`
chargé. Les paramètres Python `tracking_uri` et `registry_uri`, ou les options
CLI `--tracking-uri` et `--registry-uri`, permettent de préciser les services.

Le registre conserve le dossier ALS natif et ses quatre fichiers. Le
chargement utilise ce module du projet ; aucun modèle `pyfunc` ni service
MLflow Serving n'est créé. La future API FastAPI pourra réutiliser ce chargeur.

La version locale **1** de `movies-and-me-als` a été enregistrée avec l'alias
`champion`. Les tests de [tests/test_registry.py](tests/test_registry.py)
couvrent le chargement local et distant, l'enregistrement, les changements
d'alias, les recommandations et le rejet des modèles incohérents ou issus
d'un run inachevé. La phase 6 est terminée.

## Phase 7 : données et pipeline avec DVC

Le dépôt est initialisé avec **DVC 3.67.1**. Les dépendances pour S3 et la
connexion AWS par navigateur sont fixées dans
[requirements.txt](requirements.txt). Pour les installer dans une nouvelle
copie du projet, depuis la racine avec `.venv` activé :

```bat
python -m pip install -r requirements.txt
dvc --version
python -m pip check
```

`dvc init` a déjà été exécuté : il ne faut pas le relancer après un clonage.
Git conserve `.dvc/config`, `.dvc/.gitignore` et `.dvcignore`. Le cache et
la configuration propre à cette machine restent hors de Git.

### Retrouver les données

Les quatre CSV MovieLens restent dans `data/raw/`, avec le même contenu.
Ils sont suivis par leurs fichiers `.dvc` : `movies.csv.dvc`,
`ratings.csv.dvc`, `links.csv.dvc` et `tags.csv.dvc`. Git conserve ces
descriptions et la [source et licence](data/raw/README.md), tandis que DVC
conserve les CSV dans son cache et dans le stockage configuré. Les anciennes
versions des CSV restent présentes dans l'historique Git.

Le stockage partagé est **Amazon S3**, dans le bucket `movies-and-me`, en
région Stockholm (`eu-north-1`). Le remote `cloud` désigne
`s3://movies-and-me/movies-and-me-recommender`. Son adresse et sa région
sont enregistrées dans `.dvc/config`, suivi par Git. C'est le stockage
par défaut du projet et de ce PC. L'envoi et la récupération depuis S3 ont
été vérifiés.

Pour l'authentification interactive, installer
[AWS CLI version 2](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html),
puis ouvrir un nouveau terminal. `aws login`, disponible à partir de la
version 2.32.0, permet de réutiliser la connexion à la console AWS dans le
navigateur. Pour une identité IAM, l'accès local demande la permission
`SignInLocalDevelopmentAccess`. Si le compte utilise IAM Identity Center,
configurer plutôt la connexion avec `aws configure sso`.
[Connexion AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-sign-in.html),
[connexion avec IAM Identity Center](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-sso.html).

Les commandes suivantes créent le profil `movies-and-me`, vérifient la
connexion et le choisissent pour le remote DVC :

```bat
aws --version
aws configure set region eu-north-1 --profile movies-and-me
aws login --profile movies-and-me
aws sts get-caller-identity --profile movies-and-me
dvc remote modify --local cloud profile movies-and-me
dvc pull -r cloud
dvc status
```

La connexion se termine dans le navigateur. Le nom du profil est enregistré
dans `.dvc/config.local` ; les identifiants temporaires et les sessions
restent dans la configuration AWS, hors du dépôt. Lorsque la session
expire, relancer `aws login --profile movies-and-me` avant le transfert.

La commande `pull` récupère les CSV et les sorties du pipeline pour la
version Git courante. Le code et les fichiers de configuration viennent
de Git. [Documentation DVC](https://doc.dvc.org/command-reference/pull).

Le dossier local utilisé pour les premiers essais reste disponible sur ce
PC. Dans une nouvelle copie sur ce même PC, il peut être ajouté en option :

```bat
dvc remote add --local local C:/dev/movies-and-me-recommender-dvc-storage
dvc pull -r local
```

Son adresse reste dans `.dvc/config.local`, hors de Git. Cette option ne
change pas le stockage par défaut ; les transferts vers S3 utilisent
`-r cloud`.
[DVC avec Amazon S3](https://doc.dvc.org/user-guide/data-management/remote-storage/amazon-s3).

### Relancer le pipeline

[dvc.yaml](dvc.yaml) décrit deux étapes, exécutées dans cet ordre :

| Étape | Travail effectué | Sorties |
|---|---|---|
| `prepare` | Vérifier les données et décrire leur séparation temporelle | `artifacts/dvc/evaluation.json`, `reports/dvc/data.json` |
| `train` | Entraîner les trois essais ALS, les comparer aux références et sauvegarder l'essai retenu | `artifacts/dvc/model/`, `reports/dvc/comparison.json`, `reports/dvc/metrics.json` |

Les commandes de [src/pipeline/run.py](src/pipeline/run.py) réutilisent la
préparation et l'entraînement existants. Le choix des paramètres utilise
toujours la validation ; le modèle reste entraîné sur l'apprentissage
uniquement. Le test final reste réservé à une étape ultérieure.

```bat
dvc repro
dvc metrics show
python -m src.models.registry recommend --model-uri artifacts/dvc/model --favorites 27205 157336 155 603 --n 10
```

Les paramètres sont dans [params.yaml](params.yaml) : `evaluation` conserve
le seuil positif et les deux dates de coupure ; `als` définit les essais et
la limite de recommandations ; `tracking.uri` désigne la base MLflow.
Changer un paramètre ou une dépendance de code ou de données provoque la
reprise des étapes concernées au prochain `dvc repro`.

[.gitattributes](.gitattributes) fixe les fins de ligne du code et des
paramètres à LF. Leurs empreintes restent ainsi les mêmes après un clonage
sous Windows ou Linux. Les CSV récupérés par DVC conservent leurs octets
d'origine.

Le cache DVC conserve le modèle, la configuration générée et les rapports
détaillés. [dvc.lock](dvc.lock) relie leurs empreintes aux données, au code
et aux paramètres de chaque étape. Le fichier
`reports/dvc/metrics.json` contient les métriques sans durées ni identifiants
de runs ; il est conservé dans Git pour faciliter leur comparaison. Le
rapport complet reste suivi par DVC, avec les identifiants MLflow et les
durées propres à son exécution.

Si les entrées et les sorties sont déjà à jour, `dvc repro` ne relance pas
l'entraînement et ne crée aucun run MLflow. Pour refaire les calculs :

```bat
dvc repro --force
dvc push -r cloud
git status
```

Cette relance crée de nouveaux runs et de nouvelles durées, même si les
scores sont les mêmes. Elle met à jour `dvc.lock` ; ce fichier et les
métriques sont à committer avec les changements du pipeline ou des
paramètres. [Fonctionnement de `dvc repro`](https://doc.dvc.org/command-reference/repro).

Pour relancer ces étapes, passer par DVC : il gère les sorties déclarées
dans `dvc.yaml`. Les commandes Python appelées directement refusent
d'écraser des sorties déjà présentes.

Une variable `MLFLOW_TRACKING_URI` définie dans le terminal prend le pas sur
`tracking.uri`. DVC ne surveille pas les changements de variables
d'environnement : utiliser `dvc repro --force` si seul ce choix a changé.
Le pipeline n'enregistre aucune version dans le Model Registry et ne
déplace pas l'alias `champion`. L'enregistrement reste une commande
distincte, décrite dans la phase 6.

La phase 7 est terminée. Le pipeline a été exécuté depuis un clone propre ;
une relance forcée a retrouvé les mêmes métriques et un modèle rechargeable,
sans changer `champion`. Les 216 tests passent.

La récupération S3 a ensuite été vérifiée depuis un nouveau clone sur
**ce même PC Windows**, avec un cache vide et sans remote local. Les quatre
CSV ont les mêmes SHA-256 et le fichier de métriques est identique. Le
modèle récupéré produit dix recommandations. `dvc repro` saute les deux
étapes déjà à jour, sans créer de run MLflow ni changer les alias. Dans ce
clone, les états Git et DVC sont propres.

L'exécution sur une autre machine et dans la CI n'a pas encore été testée.
La connexion AWS de la CI sera traitée en phase 9. La phase 7 est intégrée
dans `main`.

## Phase 8 : API de recommandations avec FastAPI

La phase 8 est terminée et intégrée dans `main`. [src/api/main.py](src/api/main.py)
expose les routes HTTP ; [src/api/service.py](src/api/service.py) réutilise
le chargeur de modèle et les recommandations existantes. Les formats des
requêtes et des réponses sont définis dans
[src/api/schemas.py](src/api/schemas.py).

### Démarrer l'API

Depuis la racine du dépôt, dans **cmd**, avec `.venv` activé :

```bat
python -m pip install -r requirements.txt
set "MODEL_URI=models:/movies-and-me-als/1"
set "MLFLOW_TRACKING_URI=sqlite:///mlruns.db"
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --workers 1
```

`MODEL_URI` désigne ici la version **1** du registre local MLflow. C'est
aussi la valeur utilisée si cette variable n'est pas définie. Pour servir
une autre version, changer son numéro avant le démarrage. La base MLflow
et les fichiers du modèle doivent être accessibles au moment du chargement.
`MLFLOW_REGISTRY_URI` peut désigner un registre distinct ; sinon le registre
utilise le même service que le suivi MLflow.

L'API charge le modèle une fois au démarrage de chaque processus. Un modèle
absent ou incohérent empêche le démarrage. Cette première configuration
utilise un seul processus ; le service protège les appels simultanés au
modèle partagé. Le chargement suit le
[cycle de vie FastAPI](https://fastapi.tiangolo.com/advanced/events/).

La documentation interactive est disponible à
<http://127.0.0.1:8000/docs>. Elle permet de saisir une requête, de l'envoyer
et de consulter la réponse. `Ctrl+C` arrête le serveur.

Pour utiliser le modèle du pipeline DVC, récupérer son dossier si nécessaire,
puis choisir ce chemin avant de lancer le même serveur :

```bat
dvc pull artifacts/dvc/model -r cloud
set "MODEL_URI=artifacts/dvc/model"
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --workers 1
```

Cette récupération utilise la connexion AWS décrite en phase 7. Une fois
le dossier local chargé, les recommandations utilisent les fichiers en
mémoire : elles ne demandent ni CSV, ni appel TMDB, ni connexion AWS.
Un dossier local conserve son `run_id`, mais ses champs `name` et `version`
valent `null` : le modèle du pipeline n'est pas enregistré automatiquement
dans le Model Registry.

### Appeler les routes

| Route | Rôle |
|---|---|
| `GET /health` | Répondre `200` avec `{"status":"ok"}` lorsque le modèle est prêt |
| `GET /model/info` | Retourner `name`, `version`, `run_id` et `model_uri` du modèle chargé |
| `POST /recommendations` | Classer les films à partir des favoris TMDB et retourner les recommandations |

Le corps de `POST /recommendations` est un objet JSON :

```json
{
  "favorite_movie_ids": [27205, 157336, 155, 603],
  "n": 10
}
```

`favorite_movie_ids` est obligatoire et accepte jusqu'à **500 IDs TMDB**,
chacun étant un nombre entier strictement positif. La liste peut être vide.
`n` vaut **10** par défaut et doit être un entier compris entre **1 et 50**.
Les nombres décimaux, les chaînes de caractères, les booléens et les champs
supplémentaires sont refusés, avec une réponse HTTP **422**. Une API dont le
modèle n'est pas prêt répond **503** à la vérification de disponibilité.

Dans un deuxième terminal **cmd** :

```bat
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/model/info
curl -X POST http://127.0.0.1:8000/recommendations -H "Content-Type: application/json" -d "{\"favorite_movie_ids\":[27205,157336,155,603],\"n\":10}"
```

La réponse contient trois champs :

- `recommendations` : une liste d'objets avec `tmdb_id` et `score` ;
- `method` : `als`, ou `popularity` lorsqu'aucun favori n'est connu ;
- `model` : le nom, la version exacte, le `run_id` et l'URI du modèle chargé.

Les favoris répétés comptent une seule fois et les IDs inconnus sont ignorés.
Une liste vide ou entièrement inconnue utilise la popularité apprise à
l'entraînement. Les favoris connus sont exclus des résultats. `n` fixe une
limite : la réponse peut contenir moins de films si le catalogue ne suffit
pas. Les scores servent au classement ; ils ne sont pas des probabilités.
Aucun entraînement n'est lancé pendant une requête.

### Vérifier le service et l'API

Les tests de [tests/test_recommendation_service.py](tests/test_recommendation_service.py)
et [tests/test_api.py](tests/test_api.py) utilisent de petits modèles ALS
locaux, sans accès cloud :

```bat
python -m unittest discover -s tests -p "test_recommendation_service.py" -v
python -m unittest discover -s tests -p "test_api.py" -v
```

Les 25 tests du service et de l'API passent. Ils couvrent les requêtes
normales, les listes vides, les favoris inconnus, les entrées incorrectes,
les modèles absents ou corrompus et les appels simultanés. Les résultats
correspondent aux appels directs et les facteurs entraînés restent
inchangés. La suite complète de **241 tests** passe depuis un clone propre.

Un essai HTTP avec Uvicorn et la version 1 du registre a aussi vérifié les
trois routes, dix recommandations identiques au calcul direct, le recours
à la popularité, les erreurs 422 et la page `/docs`. Il n'a créé aucun run
MLflow ni changé le registre.

Depuis un autre clone avec un cache vide sur ce PC Windows, seul le dossier
modèle a été récupéré depuis S3. L'API a répondu avec dix recommandations,
sans CSV et sans accès à un suivi ou un registre MLflow. Les sorties DVC
ont été recalculées pour les nouvelles dépendances, avec les mêmes
métriques, puis envoyées vers S3. La version `champion` est conservée.

## Phase 9 : Docker

Le `Dockerfile` construit une image CPU avec Python 3.12. L'API utilise
`requirements-api.txt`, qui contient ses dépendances et le client léger
MLflow. L'environnement complet des notebooks et de l'entraînement reste
défini dans `requirements.txt`.

L'image contient le code de l'API. Le modèle est fourni au démarrage dans
un dossier monté en lecture seule. Les CSV, la base MLflow, les notebooks,
le cache DVC et les identifiants AWS sont exclus du contexte de construction.
Uvicorn utilise un seul worker et le service s'exécute avec un utilisateur
sans droits administrateur.

### Lancer avec le modèle DVC

Ouvrir Docker Desktop avec le moteur Linux. Depuis la racine du dépôt,
dans **cmd** ou PowerShell :

```bat
docker version
dvc pull artifacts/dvc/model -r cloud
docker compose up --build -d --wait
docker compose ps
```

La récupération DVC utilise la connexion AWS du PC. Le conteneur charge
ensuite les quatre fichiers de `artifacts/dvc/model`, avec
`MODEL_URI=/models/als`. Ce dossier doit déjà exister : Compose refuse de
créer un répertoire vide à sa place. L'API fonctionne ensuite à partir
du modèle chargé en mémoire.

La documentation est disponible sur <http://127.0.0.1:8000/docs>.
Le port est exposé uniquement sur le PC local. Le contrôle `/health`
confirme que le modèle est prêt.

Dans l'environnement Python du projet, vérifier les routes et une requête
avec un film du catalogue :

```bat
python scripts/check_api.py --favorite-id 27205
```

Le script vérifie l'identité du modèle, les recommandations ALS, l'exclusion
des favoris, le recours à la popularité et les erreurs 422. Le modèle local
conserve son `run_id` ; les champs de registre `name` et `version` sont à
`null`, comme pour le chargement local décrit en phase 8.

Pour consulter les logs et arrêter le service :

```bat
docker compose logs api
docker compose down
```

### Vérifier avec un petit modèle synthétique

Le script suivant entraîne un modèle de six films dans un nouveau dossier.
Il sert aux vérifications du conteneur et de la CI :

```bat
python scripts/create_smoke_model.py --output artifacts/smoke-model
```

Le dossier de sortie doit être nouveau. Les IDs TMDB synthétiques vont de
100 à 600 ; `check_api.py` utilise le favori 100 par défaut. Le modèle réel
DVC reste dans son propre dossier.

La construction et le démarrage avec Compose ont été vérifiés sous Docker
Desktop Linux sur Windows. Avec le modèle DVC, dix recommandations
correspondent au calcul direct sous Windows. Les routes HTTP, la solution
de secours, les erreurs 422 et `/docs` répondent correctement. Le modèle et
le système de fichiers du conteneur sont en lecture seule.

## Reproduire l'analyse

Depuis la racine du dépôt, avec **Python 3.12**, sous PowerShell :

```powershell
# À la première installation uniquement.
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# Après configuration du stockage DVC décrit dans la phase 7.
.\.venv\Scripts\python.exe -m dvc pull

# Exécuter toutes les cellules et sauvegarder leurs résultats dans le notebook.
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=120 notebooks\01_exploration.ipynb
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=120 notebooks\02_content_based_baseline.ipynb
# Le rapport et le modèle ALS doivent avoir été produits par src.training.train.
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=120 notebooks\03_implicit_als.ipynb

# Tester les calculs, jointures et cas limites indépendamment du dataset réel.
.\.venv\Scripts\python.exe -m unittest discover -s tests -v

# Ouvrir l'interface pour consulter ou réexécuter le notebook.
.\.venv\Scripts\python.exe -m jupyter lab notebooks\01_exploration.ipynb
# Ou ouvrir la baseline de phase 2.
.\.venv\Scripts\python.exe -m jupyter lab notebooks\02_content_based_baseline.ipynb
# Ou consulter les essais et les recommandations ALS.
.\.venv\Scripts\python.exe -m jupyter lab notebooks\03_implicit_als.ipynb
```

Sous macOS/Linux, créer l'environnement avec `python3.12 -m venv .venv`,
remplacer `.\.venv\Scripts\python.exe` par `.venv/bin/python` et utiliser `/`
dans le chemin du notebook. Aucune activation de l'environnement n'est requise.

## Workflow Git

`main` est la branche locale stable. Chaque fonctionnalité utilise sa propre
branche, par exemple `feat/data-exploration`, `feat/content-recommender` et
`feat/id-mapping`. Créer la branche avant les modifications et committer une
étape cohérente lorsque ses tests et son notebook sont validés. Utiliser des
messages explicites, par exemple `feat: implement content-based recommender`.
