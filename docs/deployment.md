# Déploiement

L'image `deployment` contient le service et une copie du modèle DVC. Elle
démarre sans volume externe. Le modèle est vérifié pendant la construction.
La CI la publie sur GHCR, puis la déploie sur AWS Lambda derrière une URL HTTPS.

## Construire l'image

Depuis la racine du dépôt, avec Docker BuildKit et l'environnement Python activé :

```bash
dvc pull artifacts/dvc/model -r cloud
REVISION=$(git rev-parse HEAD)
docker build --target deployment \
  --build-context model=./artifacts/dvc/model \
  --build-arg VCS_REF="$REVISION" \
  --tag "movies-me-api:$REVISION" .
```

Le modèle est transmis dans un contexte séparé. L'image n'en garde que la copie
de service, produite par `python -m src.models.serving` : les facteurs des films,
le catalogue et la popularité. Les facteurs des utilisateurs d'entraînement et la
matrice ne servent pas aux requêtes, qui recalculent le profil à partir des
favoris envoyés. Pour MovieLens 32M, cette copie pèse 15 Mo au lieu de 93 Mo et
donne les mêmes recommandations. Les données brutes, les identifiants AWS et la
base MLflow restent sur la machine.

La révision du code est enregistrée dans le label
`org.opencontainers.image.revision`. Le manifeste du modèle conserve ses
métadonnées d'entraînement et son `run_id`.

## Vérifier l'image

Le port 8001 permet de conserver l'API locale sur le port 8000 :

```bash
docker run --detach --name movies-me-api-deployment \
  --publish 127.0.0.1:8001:8000 \
  --read-only --tmpfs /tmp:rw,noexec,nosuid,size=64m \
  --cap-drop ALL --security-opt no-new-privileges \
  "movies-me-api:$REVISION"
RUN_ID=$(python -c "import json; from pathlib import Path; print(json.loads(Path('artifacts/dvc/model/manifest.json').read_text())['metadata']['run_id'])")
python scripts/check_api.py --url http://127.0.0.1:8001 \
  --favorite-id 27205 --expected-run-id "$RUN_ID"
```

Le contrôle vérifie le modèle chargé, les recommandations, le repli sur la
popularité et le rejet des entrées invalides. Consulter les logs et arrêter
le conteneur après l'essai :

```bash
docker logs movies-me-api-deployment
docker rm --force movies-me-api-deployment
```

La construction par défaut et Docker Compose utilisent la cible `runtime`,
avec un modèle monté depuis le disque. Sur les pull requests, la CI construit
la cible `deployment` avec un modèle synthétique et vérifie l'API sans accès AWS.

## Image publiée sur GHCR

Après chaque push sur `main`, la CI publie l'image dans le package privé
`ghcr.io/samirmaoude/movies-and-me-recommender`. Le job `publish` démarre
seulement si les tests et le test Docker ont réussi. Il récupère le modèle DVC
depuis S3, construit la cible `deployment`, vérifie l'API et le `run_id` du
modèle, puis pousse cette même image. Son tag est le commit Git complet. Le
résumé de l'exécution indique le digest et le `run_id`. Le job `check-published`
télécharge ensuite l'image par son digest et refait les vérifications.
Les pull requests ne publient rien.

Avant de pousser, le job `publish` vérifie que le package existe déjà et qu'il
n'est pas lisible sans authentification. Créé par `GITHUB_TOKEN` depuis ce dépôt
public, le package est devenu public ; la CI ne le crée donc jamais elle-même.

### Récupérer l'image

Créer un jeton classique avec le seul scope `read:packages`, puis se connecter
sans laisser le jeton dans l'historique du shell :

```bash
read -rs CR_PAT
printf '%s' "$CR_PAT" | docker login ghcr.io -u SamirMaoude --password-stdin
unset CR_PAT
```

Télécharger l'image d'un commit de `main`, ou une version précise par son digest :

```bash
COMMIT=$(git rev-parse origin/main)
IMAGE="ghcr.io/samirmaoude/movies-and-me-recommender:$COMMIT"
docker pull "$IMAGE"
# ou : docker pull ghcr.io/samirmaoude/movies-and-me-recommender@sha256:<digest>
docker image inspect --format '{{index .Config.Labels "org.opencontainers.image.revision"}}' "$IMAGE"
```

### Vérifier l'image publiée

Le `run_id` attendu figure dans le résumé de l'exécution. Avec le modèle DVC
de la même révision, il se lit aussi dans son manifeste :

```bash
RUN_ID=$(python -c "import json; print(json.load(open('artifacts/dvc/model/manifest.json'))['metadata']['run_id'])")
docker run --detach --name movies-me-api-ghcr \
  --publish 127.0.0.1:8002:8000 \
  --read-only --tmpfs /tmp:rw,noexec,nosuid,size=64m \
  --cap-drop ALL --security-opt no-new-privileges \
  "$IMAGE"
curl -s http://127.0.0.1:8002/model/info
python scripts/check_api.py --url http://127.0.0.1:8002 \
  --favorite-id 27205 --expected-run-id "$RUN_ID"
docker rm --force movies-me-api-ghcr
docker logout ghcr.io
```

### Créer le package privé

Si le package doit être recréé, y pousser une première fois depuis une machine
l'image construite et vérifiée plus haut, avec un jeton classique limité à
`write:packages`. Un package créé ainsi est privé :

```bash
read -rs CR_PAT
printf '%s' "$CR_PAT" | docker login ghcr.io -u SamirMaoude --password-stdin
unset CR_PAT
REVISION=$(git rev-parse HEAD)
docker tag "movies-me-api:$REVISION" "ghcr.io/samirmaoude/movies-and-me-recommender:$REVISION"
docker push "ghcr.io/samirmaoude/movies-and-me-recommender:$REVISION"
docker logout ghcr.io
```

Dans les paramètres du package, ajouter ensuite le dépôt dans
**Manage Actions access** avec le rôle **Write**. Une requête anonyme doit
être refusée :

```bash
curl -s -o /dev/null -w '%{http_code}\n' \
  https://github.com/users/SamirMaoude/packages/container/package/movies-and-me-recommender
```

La réponse attendue est `404`. Un package public ne peut plus redevenir privé :
il faut alors le supprimer, puis le recréer de cette manière.

### Accès utilisés par la publication

- Environnement GitHub `ghcr` : limité à la branche `main`, il contient le
  secret `AWS_ROLE_ARN`.
- Package GHCR : privé. Seul le job `publish` reçoit `packages: write` ;
  `check-published` lit l'image avec `packages: read`. Le dépôt doit avoir le
  rôle **Write** dans **Manage Actions access**.
- AWS : un fournisseur OIDC `token.actions.githubusercontent.com` (audience
  `sts.amazonaws.com`) et le rôle `movies-and-me-recommender-dvc-read`.
  Ce rôle autorise seulement `s3:GetObject` sur
  `movies-and-me/movies-and-me-recommender/*` et `s3:ListBucket` sur ce préfixe.

Ce dépôt utilise le format immuable du sujet OIDC, qui contient les identifiants
du compte et du dépôt GitHub. La politique de confiance du rôle impose :

```json
"StringEquals": {
  "token.actions.githubusercontent.com:aud": "sts.amazonaws.com",
  "token.actions.githubusercontent.com:sub": "repo:SamirMaoude@55006867/movies-and-me-recommender@1405677429:environment:ghcr",
  "token.actions.githubusercontent.com:ref": "refs/heads/main"
}
```

Le format réel du sujet se lit avec
`GET /repos/SamirMaoude/movies-and-me-recommender/actions/oidc/customization/sub`.

## API HTTPS sur AWS Lambda

L'API publique tourne sur AWS Lambda, dans la région `eu-north-1`, avec l'image
publiée sur GHCR : <https://7xmkh4737vk2yl5ribrhs5tzei0azgje.lambda-url.eu-north-1.on.aws/>

```bash
API=https://7xmkh4737vk2yl5ribrhs5tzei0azgje.lambda-url.eu-north-1.on.aws
curl -s "$API/version"
curl -s -X POST "$API/recommendations" \
  -H "Content-Type: application/json" \
  -d '{"favorite_movie_ids":[27205,157336,155,603],"n":10}'
```

`/version` donne le commit et le modèle servis. L'extension Lambda Web Adapter,
incluse dans l'image, transmet les requêtes à Uvicorn. Après une période sans
trafic, la première requête démarre une instance en 2 à 3 secondes.

### Déploiement automatique

Le job `deploy` suit `check-published` et ne tourne donc que sur `main`. Il :

1. copie l'image de GHCR dans le dépôt ECR `movies-and-me-recommender`, avec le
   commit comme tag, et vérifie que le digest n'a pas changé ;
2. publie une nouvelle version de la fonction Lambda, dont la description donne
   le commit et le `run_id` du modèle ;
3. dirige l'alias `staging` vers cette version et vérifie son URL ;
4. dirige l'alias `live` vers la même version et refait la vérification. Si elle
   échoue, `live` revient à la version précédente et le job échoue.

La vérification lance `scripts/check_api.py` sur l'URL publique : disponibilité,
commit, `run_id`, recommandations ALS, repli sur la popularité et rejet des
entrées invalides. Le résumé de l'exécution indique les versions servies et les
dernières versions publiées.

L'alias `staging` sert d'environnement de validation :
<https://km3ssssewokd3jq4rh3kfavmni0ionmc.lambda-url.eu-north-1.on.aws/>. Les
deux alias appartiennent à la même fonction et partagent ses limites.

Ces étapes se trouvent dans `scripts/deploy_lambda.sh`. Pour voir l'état depuis
un terminal disposant d'un accès AWS administrateur :

```bash
export AWS_PROFILE=movies-and-me AWS_REGION=eu-north-1
scripts/deploy_lambda.sh status
```

### Revenir à une version précédente

Dans l'onglet **Actions** de GitHub, lancer le workflow **Roll back** sur `main`
avec le numéro de la version voulue, indiqué dans le résumé du déploiement. Le
workflow dirige `live` vers cette version, puis vérifie que l'URL publique sert
le commit et le `run_id` inscrits dans sa description. Relancer le workflow avec
le numéro de la dernière version permet d'y revenir.

Depuis un terminal disposant d'un accès AWS administrateur, la commande
équivalente est :

```bash
scripts/deploy_lambda.sh promote 3
```

Le changement d'alias est immédiat : la version choisie reprend son image et
son modèle. Elle reste utilisable tant que son image est dans ECR, qui garde les
10 images les plus récentes.

### Limites

- La concurrence réservée limite la fonction à 2 instances. Une instance traite
  une requête à la fois ; au-delà, Lambda répond `429 Too Many Requests`.
- Chaque instance dispose de 1 024 Mo et d'une part de CPU proportionnelle. Avec
  le bundle complet du modèle MovieLens 32M, l'API utilisait 582 Mo et démarrait
  en 5,5 s ; la copie de service réduit le modèle chargé de 191 Mo à 26 Mo en
  mesure locale. Une requête est interrompue après 15 secondes.
- Une requête accepte au plus 500 favoris et `n` vaut au plus 50.
- Les URL ne demandent pas d'authentification. Seuls `staging` et `live` en ont
  une, et les permissions publiques n'autorisent l'invocation que par ces URL.
- Le rôle de déploiement ne peut changer ni la configuration de la fonction, ni
  sa concurrence, ni ses URL, et ne peut pas supprimer d'image.
- Le budget AWS `movies-and-me-monthly` (5 USD par mois) envoie un e-mail à 80 %
  puis à 100 % des dépenses réelles.

Pour couper l'API en cas d'abus, puis la rétablir :

```bash
aws lambda put-function-concurrency --function-name movies-and-me-recommender \
  --reserved-concurrent-executions 0
aws lambda put-function-concurrency --function-name movies-and-me-recommender \
  --reserved-concurrent-executions 2
```

### Coûts

Un usage modéré reste dans l'offre gratuite permanente de Lambda : un million de
requêtes et 400 000 Go-secondes par mois. Les 10 images conservées dans ECR
coûtent environ 0,10 USD par Go et par mois. Les logs sont gardés 14 jours.

Au-delà, Lambda facture 0,20 USD par million de requêtes, plus le temps de
calcul. Si les deux instances restaient occupées tout le mois, le calcul
coûterait au plus 81 USD à 1 024 Mo, plus les requêtes. Le budget le signale
dans la journée, et la commande ci-dessus arrête la dépense.

### Ressources AWS

- Dépôt ECR `movies-and-me-recommender` : tags immuables, analyse des images à
  l'envoi, conservation des 10 images les plus récentes. Sa politique autorise
  le service Lambda à lire les images pour les fonctions
  `movies-and-me-recommender*` du compte.
- Fonction Lambda `movies-and-me-recommender` : image x86_64, 1 024 Mo, délai de
  15 secondes, concurrence réservée de 2. Son rôle d'exécution,
  `movies-and-me-recommender-lambda`, peut seulement écrire dans le groupe de
  logs `/aws/lambda/movies-and-me-recommender`.
- Alias `staging` et `live`, chacun avec une URL de fonction sans
  authentification.
- Rôle `movies-and-me-recommender-deploy`, utilisé par le job `deploy` et par
  le workflow **Roll back**. Il copie les images dans le dépôt ECR, met à jour
  le code de la fonction, publie des versions et déplace les alias.
- Environnement GitHub `production` : limité à `main`, il contient le secret
  `AWS_DEPLOY_ROLE_ARN`.
- Budget AWS `movies-and-me-monthly`.

La politique de confiance du rôle de déploiement impose :

```json
"StringEquals": {
  "token.actions.githubusercontent.com:aud": "sts.amazonaws.com",
  "token.actions.githubusercontent.com:sub": "repo:SamirMaoude@55006867/movies-and-me-recommender@1405677429:environment:production",
  "token.actions.githubusercontent.com:ref": "refs/heads/main"
}
```

Une URL publique demande deux permissions par alias. Si une URL doit être
recréée :

```bash
for alias in staging live; do
  aws lambda create-function-url-config --function-name movies-and-me-recommender \
    --qualifier "$alias" --auth-type NONE
  aws lambda add-permission --function-name movies-and-me-recommender \
    --qualifier "$alias" --statement-id "public-url-$alias" \
    --action lambda:InvokeFunctionUrl --principal "*" --function-url-auth-type NONE
  aws lambda add-permission --function-name movies-and-me-recommender \
    --qualifier "$alias" --statement-id "public-invoke-$alias" \
    --action lambda:InvokeFunction --principal "*" --invoked-via-function-url
done
```

Une URL recréée change d'adresse : mettre alors à jour ce document et le README.
