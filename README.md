# Movies & Me Recommender

Service de recommandations pour l'application Movies-And-Me. L'API reçoit
une liste de favoris TMDB et renvoie des films classés par le modèle ALS
d'`implicit`. L'entraînement utilise MovieLens ; DVC versionne les données
et les modèles, et MLflow suit les essais et les versions enregistrées.

L'application fournit des favoris, pas des notes. Les notes MovieLens
supérieures ou égales à 4/5 servent de favoris simulés pour l'entraînement
et l'évaluation. La popularité et les genres servent de références de comparaison.

## Installation

Les commandes Windows ci-dessous s'exécutent depuis la racine du dépôt,
dans **cmd**, avec Python 3.12 :

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pip check
```

Sous PowerShell, activer l'environnement avec
`.\.venv\Scripts\Activate.ps1`. Sous macOS/Linux, le créer avec
`python3.12 -m venv .venv`, puis l'activer avec `source .venv/bin/activate`.

| Fichier | Environnement |
|---|---|
| [requirements.txt](requirements.txt) | Notebooks, entraînement, DVC, MLflow et API |
| [requirements-api.txt](requirements-api.txt) | API et client MLflow léger, utilisés par l'image Docker |
| [requirements-ci.txt](requirements-ci.txt) | Tests et contrôles Python de la CI |

## Lancer l'API

Récupérer le modèle DVC après avoir configuré l'[accès S3](#données-et-stockage-dvc) :

```bat
dvc pull artifacts/dvc/model -r cloud
set "MODEL_URI=artifacts/dvc/model"
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --workers 1
```

Sous PowerShell, définir la variable avec
`$env:MODEL_URI = "artifacts/dvc/model"`. Sous macOS/Linux, utiliser
`export MODEL_URI=artifacts/dvc/model`.

La documentation interactive est disponible à <http://127.0.0.1:8000/docs>.
Le modèle est chargé une fois au démarrage ; un dossier absent ou incohérent
empêche le démarrage. Une fois chargé, il n'a besoin ni des CSV, ni d'AWS,
ni de l'API TMDB pour répondre. Aucun entraînement n'est lancé pendant
une requête. `Ctrl+C` arrête le serveur.

`MODEL_URI` accepte aussi une URI MLflow :

```bat
set "MODEL_URI=models:/movies-and-me-als/1"
set "MLFLOW_TRACKING_URI=sqlite:///mlruns.db"
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --workers 1
```

`models:/movies-and-me-als/1` est la valeur par défaut de `MODEL_URI`.
Cette version doit exister dans le registre choisi. `MLFLOW_REGISTRY_URI`
permet de désigner un registre distinct ; sinon il utilise le même service
que le suivi MLflow. Un alias comme `models:/movies-and-me-als@champion`
est résolu au chargement et le service conserve cette version jusqu'à son redémarrage.

### Requêtes et réponses

| Route | Réponse |
|---|---|
| `GET /health` | `200` avec `{"status":"ok"}` quand le modèle est prêt ; `503` sinon |
| `GET /model/info` | Nom, version, `run_id` et URI du modèle chargé |
| `POST /recommendations` | Films recommandés à partir des favoris TMDB |

Corps de `POST /recommendations` :

```json
{
  "favorite_movie_ids": [27205, 157336, 155, 603],
  "n": 10
}
```

`favorite_movie_ids` est obligatoire et accepte une liste vide ou jusqu'à
500 IDs TMDB, tous entiers et strictement positifs. `n` vaut 10 par défaut
et doit être compris entre 1 et 50. Les valeurs incorrectes et les champs
supplémentaires donnent une erreur HTTP 422.

Exemple depuis un deuxième terminal **cmd** :

```bat
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/model/info
curl -X POST http://127.0.0.1:8000/recommendations -H "Content-Type: application/json" -d "{\"favorite_movie_ids\":[27205,157336,155,603],\"n\":10}"
```

La réponse contient `recommendations` (objets avec `tmdb_id` et `score`),
`method` (`als` ou `popularity`) et `model` (identité du modèle chargé).
Pour un dossier local, `name` et `version` valent `null` ; le `run_id`
enregistré dans le modèle est conservé.

Les doublons parmi les favoris ne comptent qu'une fois. Les IDs inconnus
sont ignorés ; sans favori connu, le service renvoie la popularité apprise
à l'entraînement. Les favoris sont exclus des résultats. La réponse peut
contenir moins de `n` films si le catalogue ne suffit pas. Les scores servent
au classement et ne sont pas des probabilités.

## Docker

Démarrer Docker Desktop avec le moteur Linux, puis lancer :

```bat
docker version
dvc pull artifacts/dvc/model -r cloud
docker compose up --build -d --wait
docker compose ps
```

L'API est accessible sur <http://127.0.0.1:8000/docs>. Compose monte
`artifacts/dvc/model` en lecture seule dans `/models/als` ; ce dossier doit
exister avant le démarrage. L'image CPU utilise Python 3.12 et un seul
worker Uvicorn. Le conteneur s'exécute sans droits administrateur, avec
un système de fichiers en lecture seule, et expose le port uniquement en local.

Les CSV, la base MLflow, les notebooks, le cache DVC et les identifiants AWS
sont exclus du contexte de construction. La connexion AWS du PC sert
uniquement à récupérer le modèle avant le démarrage.

Vérifier l'API avec un film du catalogue, puis consulter les logs ou arrêter
le service :

```bat
python scripts/check_api.py --favorite-id 27205
docker compose logs api
docker compose down
```

Le script contrôle l'identité du modèle, les recommandations ALS, l'exclusion
des favoris, le classement par popularité et les erreurs 422.

Pour les tests Docker sans MovieLens ni S3, le script suivant crée un petit
modèle synthétique dans un nouveau dossier :

```bat
python scripts/create_smoke_model.py --output artifacts/smoke-model
```

Ses six films utilisent les IDs TMDB de 100 à 600. `check_api.py` utilise
le favori 100 par défaut. La CI monte ce modèle dans son conteneur de test ;
le fichier Compose du dépôt utilise le modèle DVC.

## Données et stockage DVC

Le projet utilise MovieLens `ml-latest-small`, généré le 26 septembre 2018 :
9 742 films, 100 836 notes, 610 utilisateurs et 3 683 tags. Ces utilisateurs
appartiennent au jeu MovieLens. Ils ne proviennent pas de Movies-And-Me.
La [notice MovieLens](data/raw/README.md) contient la description des fichiers,
la licence et la citation à conserver avec les données.

Les quatre CSV se trouvent dans `data/raw/` : `movies.csv`, `ratings.csv`,
`links.csv` et `tags.csv`. DVC les suit avec leurs fichiers `.dvc`.

Le remote `cloud`, défini dans [.dvc/config](.dvc/config), utilise
`s3://movies-and-me/movies-and-me-recommender` en région `eu-north-1`.
Il faut disposer d'un accès au bucket pour récupérer les fichiers.

Avec [AWS CLI version 2](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)
et un compte autorisé à utiliser la connexion par navigateur :

```bat
aws --version
aws configure set region eu-north-1 --profile movies-and-me
aws login --profile movies-and-me
aws sts get-caller-identity --profile movies-and-me
dvc remote modify --local cloud profile movies-and-me
dvc pull -r cloud
dvc status
```

`aws login` demande AWS CLI 2.32.0 ou une version ultérieure. Pour une
identité IAM, la connexion locale demande la permission
`SignInLocalDevelopmentAccess`. Avec IAM Identity Center, utiliser plutôt
`aws configure sso`. Voir la [connexion AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-sign-in.html)
et la [configuration SSO](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-sso.html).

Le profil DVC est enregistré dans `.dvc/config.local`, ignoré par Git.
Les sessions et les identifiants restent dans la configuration AWS.
Quand la session expire, relancer `aws login --profile movies-and-me`.
`dvc pull` récupère les données et les sorties correspondant à la version
Git courante ; le code et les paramètres viennent de Git.

Un stockage local peut aussi être ajouté sur une machine qui possède une
copie du cache, en remplaçant le chemin par celui de ce stockage :

```bat
dvc remote add --local local C:/dev/movies-and-me-recommender-dvc-storage
dvc pull -r local
```

### Correspondances MovieLens et TMDB

[src/data/id_mapper.py](src/data/id_mapper.py) relie les IDs MovieLens et TMDB
à partir de `links.csv`. Les films sans ID TMDB sont ignorés.

Lorsqu'un ID TMDB correspond à plusieurs films MovieLens, le modèle choisit
une seule entrée. Toutes sont exclues si cet ID figure parmi les favoris.

Les notebooks et le modèle par genres peuvent utiliser des `movieId`
MovieLens. L'API et `ImplicitRecommender` utilisent directement les IDs TMDB.

## Entraînement et pipeline

[dvc.yaml](dvc.yaml) décrit la préparation des données et l'entraînement :

| Étape | Sorties |
|---|---|
| `prepare` | `artifacts/dvc/evaluation.json`, `reports/dvc/data.json` |
| `train` | `artifacts/dvc/model/`, `reports/dvc/comparison.json`, `reports/dvc/metrics.json` |

```bat
dvc repro
dvc metrics show
python -m src.models.registry recommend --model-uri artifacts/dvc/model --favorites 27205 157336 155 603 --n 10
```

[params.yaml](params.yaml) définit le seuil positif, les dates de séparation,
les trois configurations ALS et l'adresse du suivi MLflow. Le pipeline
réutilise la préparation et l'entraînement de `src/`. Il choisit l'essai
avec le meilleur NDCG@10 de validation, puis le Recall@10 en cas d'égalité,
puis l'ordre de déclaration.

[dvc.lock](dvc.lock) relie les sorties au code, aux paramètres et aux données.
Les métriques restent dans Git ; le modèle et les rapports détaillés sont
stockés par DVC. Si les entrées et sorties sont à jour, `dvc repro` ne lance
aucun calcul ni run MLflow. Pour forcer les calculs et envoyer les sorties :

```bat
dvc repro --force
dvc push -r cloud
git status
```

Cette relance crée de nouveaux runs et met à jour `dvc.lock`. Le verrou et
les métriques doivent être commités avec les changements du pipeline.
Une variable `MLFLOW_TRACKING_URI` définie dans le terminal prend le pas
sur `tracking.uri` ; DVC ne surveille pas ce changement et demande alors
une relance forcée. Le pipeline n'enregistre aucune version dans le registre
et ne déplace pas l'alias `champion`.

Pour une expérience indépendante du pipeline, les configurations sont
dans [config/evaluation.json](config/evaluation.json) et
[config/als.json](config/als.json) :

```bat
set "MLFLOW_TRACKING_URI=sqlite:///mlruns.db"
python -m src.training.train
```

La commande produit `reports/als_comparison.json` et sauvegarde le modèle
retenu dans `artifacts/implicit-als/`. Elle refuse d'écraser un dossier de
modèle existant. Pour un nouvel essai, choisir des chemins libres :

```bat
python -m src.training.train --artifact-dir artifacts/implicit-als-run-2 --output reports/als_comparison-run-2.json
```

Le modèle contient quatre fichiers : `model.npz`, `interactions.npz`,
`catalog.json` et `manifest.json`. Le chargement vérifie leur cohérence et
leurs empreintes. `ImplicitRecommender.load("artifacts/implicit-als")`
permet ensuite de recommander en Python sans relire les CSV.

## Suivi et registre MLflow

Chaque configuration ALS crée un run dans l'expérience
`movies-and-me-recommender`. Le run contient les paramètres, les métriques,
la durée d'entraînement, `evaluation.json`, `provenance.json` et le dossier
`model/`. Le run retenu contient aussi `comparison.json`.

`provenance.json` conserve le commit Git, la présence de modifications
locales (`dirty`), les SHA-256 des données et les versions de Python et des
dépendances. Si `dirty` vaut `true`, le commit ne suffit pas à reproduire
le code exécuté : les modifications locales ne sont pas archivées.
Les durées et les identifiants des runs varient entre deux exécutions.

Ouvrir l'interface locale dans un autre terminal, avec `.venv` activé :

```bat
python -m mlflow server --backend-store-uri sqlite:///mlruns.db --host 127.0.0.1 --port 5000 --workers 1
```

L'interface est disponible à <http://127.0.0.1:5000>.
La base `mlruns.db` et les fichiers MLflow locaux sont ignorés par Git.

### Enregistrer et charger un modèle

Remplacer `RUN_ID_DU_RAPPORT` par `selection.selected_run_id` du rapport
d'entraînement, en gardant le même `MLFLOW_TRACKING_URI` :

```bat
python -m src.models.registry register --run-id RUN_ID_DU_RAPPORT --name movies-and-me-als --alias champion
python -m src.models.registry recommend --model-uri models:/movies-and-me-als@champion --favorites 27205 157336 155 603 --n 10
```

L'enregistrement utilise les fichiers du run et ne relance pas l'entraînement.
Relancer la commande pour le même run et le même nom réutilise sa version.
Sans `--alias champion`, la nouvelle version reste candidate.

L'option `--alias champion` déplace l'alias après rechargement et comparaison
des recommandations avec le modèle source. Les IDs doivent correspondre ;
les scores sont comparés avec `rtol=1e-6`, `atol=1e-7`. Cette vérification
contrôle le rechargement, pas la qualité du modèle.

[load_recommender](src/models/registry.py) accepte un dossier local,
`runs:/RUN_ID/model`, une version `models:/movies-and-me-als/1` ou un alias.
Les options CLI `--tracking-uri` et `--registry-uri` permettent de choisir
les services MLflow. Le chargeur utilise les fichiers ALS natifs.

## Évaluation

Les interactions sont séparées dans le temps, avec les mêmes dates pour
tous les utilisateurs :

| Groupe | Période UTC | Interactions | Avis positifs |
|---|---|---:|---:|
| Apprentissage | Avant le 22 mars 2016 à 08:26:11 | 80 657 | 39 251 |
| Validation | Du 22 mars 2016 à 08:26:11, avant le 21 juin 2017 à 07:17:25 | 10 082 | 5 197 |
| Test | À partir du 21 juin 2017 à 07:17:25 | 10 083 | 4 127 |

Une interaction à la date de coupure appartient au groupe suivant. Les
notes positives valent toutes 1 pour ALS ; une note de 5/5 ne pèse pas
davantage qu'une note de 4/5. La matrice d'apprentissage comprend
521 utilisateurs MovieLens et 5 223 films TMDB.

Les trois modèles reçoivent les mêmes favoris et peuvent recommander le
même catalogue. Seuls les favoris transmis sont exclus. Recall@10 mesure
la part de cibles retrouvées dans les dix premiers résultats ; NDCG@10
prend aussi en compte leur position. Chaque utilisateur a le même poids
dans la moyenne.

| Modèle | Recall@10 moyen | NDCG@10 moyen |
|---|---:|---:|
| Popularité | 4,57 % | 0,03894 |
| Genres | 0 % | 0 |
| ALS retenu, 64 facteurs | 4,81 % | 0,06231 |

Ces résultats portent sur 20 utilisateurs et 333 cibles connues, soit
6,4 % des avis positifs de validation. En comptant aussi leurs 77 cibles
hors catalogue, le Recall d'ALS descend à 3,82 %, contre 4,16 % pour la
popularité ; son NDCG reste supérieur, à 0,05612 contre 0,03481.
Ce petit groupe ne permet pas de conclure sur tous les utilisateurs de
l'application. La qualité des réponses sans favori connu n'est pas mesurée
par cette comparaison.

Le modèle sauvegardé est entraîné uniquement sur le groupe d'apprentissage.
Les scores présentés proviennent de la validation ; le jeu de test n'a pas
été utilisé.

Les rapports détaillent le protocole, les résultats et les limites :

- [Popularité et genres](reports/baseline_comparison.md), avec les [valeurs exactes](reports/baseline_comparison.json).
- [Essais ALS](reports/als_comparison.md), avec les [valeurs exactes](reports/als_comparison.json).

Pour reproduire les rapports de préparation et de comparaison des références :

```bat
python -m src.evaluation.prepare
python -m src.evaluation.compare
```

## Notebooks

| Notebook | Contenu |
|---|---|
| [01_exploration.ipynb](notebooks/01_exploration.ipynb) | Qualité des données, activité et couverture TMDB |
| [02_content_based_baseline.ipynb](notebooks/02_content_based_baseline.ipynb) | TF-IDF des genres et recommandations avec IDs MovieLens ou TMDB |
| [03_implicit_als.ipynb](notebooks/03_implicit_als.ipynb) | Essais ALS, profils TMDB et sauvegarde du modèle |

Après récupération des CSV avec DVC, ouvrir les notebooks avec :

```bat
python -m jupyter lab notebooks
```

Pour les exécuter et sauvegarder les sorties :

```bat
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=120 notebooks\01_exploration.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=120 notebooks\02_content_based_baseline.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=120 notebooks\03_implicit_als.ipynb
```

Le notebook ALS attend `reports/als_comparison.json` et
`artifacts/implicit-als/`, produits par `python -m src.training.train`.
Sous macOS/Linux, utiliser `/` dans les chemins des notebooks.

## Tests et intégration continue

La suite utilise des données synthétiques, des dossiers temporaires et des
bases MLflow SQLite de test. Elle ne demande ni CSV MovieLens ni connexion AWS.

```bat
python -m unittest discover -s tests -v
```

[.github/workflows/ci.yml](.github/workflows/ci.yml) s'exécute sur les pull
requests vers `main`, les pushes sur `main`, `chore/docker` et
`ci/github-actions`, ou manuellement. Il comporte deux jobs sous Ubuntu
avec Python 3.12 :

1. Installation et contrôle des dépendances, vérification du code et tests unitaires.
2. Entraînement du modèle synthétique, construction de l'image Docker et test HTTP de l'API.

Le second job attend que le premier passe. Le conteneur de test est supprimé
à la fin, y compris en cas d'échec. Les actions GitHub utilisent des commits
précis, avec un accès en lecture au dépôt. Le workflow ne publie pas d'image.

Pour installer les dépendances de la CI et lancer les tests :

```bat
python -m pip install -r requirements-ci.txt
python -m pip check
python -m unittest discover -s tests -v
```
