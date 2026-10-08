# Movies & Me Recommender

Backend/ML séparé de l'application Movies-And-Me. L'objectif est de construire
progressivement un recommender dont les résultats pourront être exposés avec
des identifiants TMDB.

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
de qualité. La baseline fonctionne, mais sa qualité prédictive reste à évaluer.

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

La prochaine étape est la **phase 4 : évaluation**, pour comparer cette
baseline à un classement par popularité avec Recall@10 et NDCG@10.

## Reproduire l'analyse

Depuis la racine du dépôt, avec **Python 3.12**, sous PowerShell :

```powershell
# À la première installation uniquement.
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# Exécuter toutes les cellules et sauvegarder leurs résultats dans le notebook.
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=120 notebooks\01_exploration.ipynb
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=120 notebooks\02_content_based_baseline.ipynb

# Tester les calculs, jointures et cas limites indépendamment du dataset réel.
.\.venv\Scripts\python.exe -m unittest discover -s tests -v

# Ouvrir l'interface pour consulter ou réexécuter le notebook.
.\.venv\Scripts\python.exe -m jupyter lab notebooks\01_exploration.ipynb
# Ou ouvrir la baseline de phase 2.
.\.venv\Scripts\python.exe -m jupyter lab notebooks\02_content_based_baseline.ipynb
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
