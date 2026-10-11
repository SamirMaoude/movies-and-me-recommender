# Movies & Me Recommender

API de recommandations pour Movies-And-Me, basée sur ALS avec `implicit`.
Elle reçoit les favoris sous forme d'IDs TMDB et renvoie des films recommandés.
L'entraînement utilise MovieLens : les notes d'au moins 4/5 simulent des favoris.
DVC gère les données et les modèles ; MLflow suit les entraînements et leurs versions.

## Installation

Python 3.12 est requis. Depuis la racine du dépôt :

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Récupérer les données et le modèle

Le stockage DVC est sur S3 et nécessite un accès AWS autorisé.
Avec AWS CLI 2.32 ou plus récent, configurer le profil puis récupérer les fichiers :

```bash
aws configure set region eu-north-1 --profile movies-and-me
aws login --profile movies-and-me
dvc remote modify --local cloud profile movies-and-me
dvc pull -r cloud
```

Le stockage est défini dans [.dvc/config](.dvc/config). Les CSV sont récupérés
sous `data/raw/` et le modèle sous `artifacts/dvc/model/`.

## Lancer l'API

```bash
export MODEL_URI="artifacts/dvc/model"
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --workers 1
```

Documentation interactive : <http://127.0.0.1:8000/docs>.

| Route | Usage |
|---|---|
| `GET /health` | Vérifier que le modèle est prêt |
| `GET /model/info` | Identifier le modèle chargé |
| `POST /recommendations` | Obtenir des recommandations |

Exemple depuis un autre terminal :

```bash
curl -X POST http://127.0.0.1:8000/recommendations \
  -H "Content-Type: application/json" \
  -d '{"favorite_movie_ids":[27205,157336,155,603],"n":10}'
```

`favorite_movie_ids` accepte jusqu'à 500 IDs TMDB positifs. `n` vaut 10 par défaut,
avec une limite de 50. La réponse contient les films (`tmdb_id`, `score`),
la méthode (`als` ou `popularity`) et l'identité du modèle.
Les favoris sont exclus des résultats. Sans favori connu, le service utilise
la popularité. Les scores servent au classement, ce ne sont pas des probabilités.

## Docker

Avec Docker et le modèle DVC déjà récupéré :

```bash
docker compose up --build -d --wait
python scripts/check_api.py --favorite-id 27205
```

L'API utilise le même port 8000 que le lancement natif : lancer un seul des deux.
`docker compose logs api` affiche les logs ; `docker compose down` arrête le service.
Le modèle est monté depuis `artifacts/dvc/model/`.

Pour construire une image avec le modèle inclus ou récupérer l'image publiée
sur GHCR, suivre les [commandes de déploiement](docs/deployment.md).

## Entraînement

Modifier [params.yaml](params.yaml), puis exécuter le pipeline :

```bash
dvc repro
dvc metrics show
```

Le pipeline prépare les données, compare les configurations ALS et sauvegarde
le modèle retenu dans `artifacts/dvc/model/`. Il le sélectionne sur le NDCG
de validation. Le rapport est dans `reports/dvc/comparison.json`.

Pour partager les nouvelles sorties, exécuter `dvc push -r cloud` et commiter
`params.yaml`, `dvc.lock` et les métriques mises à jour.

## Suivi et registre MLflow

Les entraînements sont suivis dans `mlruns.db`. Ouvrir l'interface :

```bash
python -m mlflow server --backend-store-uri sqlite:///mlruns.db --host 127.0.0.1 --port 5000 --workers 1
```

Interface : <http://127.0.0.1:5000>.

DVC ne récupère pas la base MLflow. Après un clonage, `dvc repro --force`
permet de créer ses propres runs même si le pipeline est déjà à jour.

Pour enregistrer le modèle, remplacer `RUN_ID` par `selection.selected_run_id`
de `reports/dvc/comparison.json`, puis lancer dans un autre terminal avec `.venv` activé :

```bash
export MLFLOW_TRACKING_URI="sqlite:///mlruns.db"
python -m src.models.registry register --run-id RUN_ID --name movies-and-me-als --alias champion
```

Cette commande associe l'alias `champion` à la version enregistrée.
Pour la servir avec Uvicorn, exécuter
`export MODEL_URI="models:/movies-and-me-als@champion"`, puis relancer l'API
dans le même terminal.

## Tests

```bash
python -m unittest discover -s tests -v
```

Les tests utilisent des données synthétiques et ne nécessitent pas d'accès AWS.
La [CI GitHub Actions](.github/workflows/ci.yml) exécute les tests, construit
l'image Docker et vérifie l'API avec un modèle synthétique. Sur `main`, elle publie
ensuite sur GHCR l'image vérifiée avec le modèle DVC.

## Analyses et résultats

- [Exploration des données](notebooks/01_exploration.ipynb)
- [Recommandations par genres](notebooks/02_content_based_baseline.ipynb)
- [Essais ALS](notebooks/03_implicit_als.ipynb)
- [Comparaison popularité / genres](reports/baseline_comparison.md)
- [Résultats ALS et limites de l'évaluation](reports/als_comparison.md)
- [Description et licence MovieLens](data/raw/README.md)

Pour ouvrir les notebooks : `python -m jupyter lab notebooks`.
