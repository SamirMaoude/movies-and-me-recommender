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

La prochaine étape recommandée est la **phase 2 : baseline content-based par
genres**, puis similarité cosinus et exclusion des favoris. Elle reste à
implémenter ; aucune autre phase n'est incluse ici.

## Reproduire l'analyse

Depuis la racine du dépôt, avec **Python 3.12**, sous PowerShell :

```powershell
# À la première installation uniquement.
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# Exécuter toutes les cellules et sauvegarder leurs résultats dans le notebook.
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=120 notebooks\01_exploration.ipynb

# Tester les calculs, jointures et cas limites indépendamment du dataset réel.
.\.venv\Scripts\python.exe -m unittest discover -s tests -v

# Ouvrir l'interface pour consulter ou réexécuter le notebook.
.\.venv\Scripts\python.exe -m jupyter lab notebooks\01_exploration.ipynb
```

Sous macOS/Linux, créer l'environnement avec `python3.12 -m venv .venv`,
remplacer `.\.venv\Scripts\python.exe` par `.venv/bin/python` et utiliser `/`
dans le chemin du notebook. Aucune activation de l'environnement n'est requise.
