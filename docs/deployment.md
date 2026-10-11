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
avec un modèle monté depuis le disque. La CI construit la cible `deployment`
avec un modèle synthétique et vérifie l'API sans accès AWS.
