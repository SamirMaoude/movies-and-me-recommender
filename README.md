# Movies & Me Recommender

Backend/ML séparé de l'application Movies-And-Me. Le projet utilise
le modèle ALS fourni par `implicit` pour se concentrer sur le MLOps :
entraînement reproductible, évaluation, déploiement et suivi en production.
L'API utilisera des identifiants TMDB.

La [feuille de route](ROADMAP.md) décrit cette direction. Les phases 1 à 5
sont réalisées : ALS est entraîné, comparé aux références et rechargeable
à partir de ses fichiers. La phase 6 est en cours : MLflow conserve les essais,
leurs modèles et les informations sur le code, les données et l'environnement.
Le registre de modèles reste à intégrer. Le modèle par genres reste disponible
pour les comparaisons.

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
`artifacts/` est ignoré par Git ; une nouvelle copie du dépôt doit entraîner
le modèle pour obtenir ces fichiers.

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

La phase 6 reste en cours. Les prochaines étapes sont l'enregistrement d'une
version dans le Model Registry avec un alias `champion`, puis un chargement
commun qui récupère cette version et vérifie ses recommandations.

## Reproduire l'analyse

Depuis la racine du dépôt, avec **Python 3.12**, sous PowerShell :

```powershell
# À la première installation uniquement.
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

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
