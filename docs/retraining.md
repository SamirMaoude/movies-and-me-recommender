# Réentraînement et sélection

Le workflow **Retrain** entraîne un modèle candidat, le compare au modèle en
service et ne propose le remplacement que si le candidat passe tous les
contrôles. La livraison réutilise la publication GHCR et le déploiement Lambda.

## Déclenchement

- À la main, depuis l'onglet **Actions**, sur `main`.
- Automatiquement, quand un push sur `main` modifie un fichier
  `data/raw/*.dvc`, c'est-à-dire une nouvelle version des données.

Si les données et les paramètres correspondent déjà à `dvc.lock`, le workflow
s'arrête sans entraîner : le même snapshot donnerait le même candidat. Deux
options permettent un essai volontaire :

- `trials` remplace la liste `als.trials` de `params.yaml` par une liste JSON ;
- `force` relance tout le pipeline sans changement.

Il n'y a pas de réentraînement planifié.

## Étapes

1. `dvc pull` récupère les données et le modèle en service.
2. `dvc repro` contrôle les données (étape `prepare`), entraîne les essais ALS
   sur la période d'entraînement et retient le meilleur sur la validation
   (étape `train`). Chaque essai devient un run MLflow.
3. `python -m src.training.promotion` évalue le candidat, le modèle en service
   et la popularité sur les mêmes utilisateurs de validation, ceux de la
   version des données du candidat.
4. Si un contrôle échoue, le workflow s'arrête : rien n'est poussé.
5. Sinon, `dvc push` envoie le modèle sur S3 et le workflow ouvre une pull
   request qui met à jour `dvc.lock` et `reports/dvc/metrics.json`. Le modèle
   contient son catalogue d'IDs TMDB et MovieLens.
6. Une fois la pull request fusionnée, la CI publie l'image sur GHCR et la
   déploie sur `staging`, puis sur `live`, en vérifiant le `run_id` du
   nouveau modèle. Le workflow **Roll back** permet de revenir à la version
   précédente.

La période de test finale n'est jamais lue : ni pour choisir l'essai, ni pour
comparer au modèle en service.

## Critères de sélection

Les règles sont dans la section `promotion` de `params.yaml` :

| Contrôle | Règle |
|---|---|
| Utilisateurs de validation | Au moins 20 |
| NDCG@10 | Au plus 0,025 sous le modèle en service |
| Recall@10 | Au plus 0,009 sous le modèle en service |
| Couverture du catalogue dans les top 10 | Au moins 80 % de celle du modèle en service |
| Temps de réponse p95 | Au plus le double du modèle en service, mesuré dans le même job |

Ces seuils viennent des résultats observés. La validation ne compte que
20 utilisateurs : en réentraînant la configuration en service avec cinq
graines aléatoires, NDCG@10 a varié de 0,029 à 0,062 (écart type 0,013) et
Recall@10 de 0,042 à 0,053 (écart type 0,0045). Le modèle en service
correspond à la meilleure graine. Les tolérances valent deux écarts types :
exiger un candidat meilleur que lui refuserait la plupart des réentraînements
par simple hasard. La couverture n'a varié que de 2 % et les temps de réponse
de 5 à 6 ms.

Un modèle volontairement dégradé (1 facteur, 1 itération) obtient NDCG@10
0,026, Recall@10 0,014 et une couverture de 0,6 % contre 3,1 % : il échoue sur
trois contrôles. La popularité, affichée pour référence, obtient NDCG@10 0,039.

Avec si peu d'utilisateurs de validation, ces contrôles écartent un modèle
nettement moins bon, mais ne prouvent pas qu'un candidat est meilleur.

## Suivi MLflow

Les runs de la CI sont enregistrés dans une base SQLite et des artefacts
stockés sous `s3://movies-and-me/movies-and-me-recommender/mlflow/`. Le
workflow attache la décision (`promotion.json`) et le tag `promotion` au run
du candidat, qu'il soit accepté ou refusé. Pour consulter ces runs avec un
accès AWS :

```bash
python -m pip install boto3==1.43.106
aws s3 cp s3://movies-and-me/movies-and-me-recommender/mlflow/mlflow.db mlflow-ci.db
python -m mlflow server --backend-store-uri sqlite:///mlflow-ci.db --host 127.0.0.1 --port 5001
```

Cette base est distincte de `mlruns.db`, qui garde les essais lancés en local.

## Essayer la sélection localement

Avec les données et le modèle DVC récupérés, le modèle en service peut être
comparé à un autre bundle :

```bash
python -m src.training.promotion --served artifacts/dvc/model \
  --candidate chemin/vers/le/candidat --output promotion.json
```

La commande se termine avec le code 1 si le candidat est refusé.

## Accès

- Environnement GitHub `retraining`, limité à `main`, avec le secret
  `AWS_RETRAIN_ROLE_ARN`.
- Rôle `movies-and-me-recommender-retrain` : lecture du préfixe
  `movies-and-me-recommender/` et écriture seulement dans
  `movies-and-me-recommender/files/md5/` (DVC) et
  `movies-and-me-recommender/mlflow/`. Il ne peut rien supprimer. Sa
  politique de confiance reprend celle du rôle de déploiement, avec le sujet
  `...:environment:retraining`.
- Le dépôt autorise GitHub Actions à créer des pull requests. Une pull request
  ouverte par `GITHUB_TOKEN` ne déclenche pas la CI ; les contrôles du
  workflow **Retrain** en tiennent lieu, et la CI s'exécute sur `main` après
  la fusion, avant tout déploiement.
