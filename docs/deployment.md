# Image de déploiement

L'image `deployment` contient le service et une copie du modèle DVC. Elle
démarre sans volume externe. Le modèle est vérifié pendant la construction.

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

Le modèle est transmis dans un contexte séparé. Seuls `catalog.json`,
`interactions.npz`, `manifest.json` et `model.npz` sont copiés dans l'image.
Les données brutes, les identifiants AWS et la base MLflow restent sur la machine.

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

### Accès utilisés par la publication

- Environnement GitHub `ghcr` : limité à la branche `main`, il contient le
  secret `AWS_ROLE_ARN`.
- Package GHCR : privé. Seul le job `publish` reçoit `packages: write` ;
  `check-published` lit l'image avec `packages: read`.
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
