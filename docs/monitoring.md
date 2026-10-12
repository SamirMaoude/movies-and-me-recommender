# Surveillance

L'API publiée sur AWS Lambda est suivie avec CloudWatch, dans la région
`eu-north-1`. Le tableau de bord, les alertes et la comparaison quotidienne
restent dans l'offre gratuite de CloudWatch : 5 métriques personnalisées,
6 alarmes et un tableau de bord.

Ces mesures décrivent le fonctionnement du service et les demandes reçues.
Elles ne disent pas si les recommandations plaisent : il faudrait pour cela des
retours des utilisateurs.

## Mesures

| Mesure | Source |
|---|---|
| Requêtes, erreurs 4xx et 5xx, latence de `live` | Métriques des URL de fonction : `UrlRequestCount`, `Url4xxCount`, `Url5xxCount`, `UrlRequestLatency` (espace `AWS/Lambda`, `Resource` = `movies-and-me-recommender:live`) |
| Requêtes rejetées et instances occupées | `Throttles` et `ConcurrentExecutions` de la fonction |
| Mémoire utilisée | `MaxMemoryUsed` (espace `MoviesAndMe`), lu par un filtre dans la ligne `REPORT` de chaque appel |
| Favoris par requête, favoris inconnus, repli sur la popularité, réponses vides | `Favorites`, `UnknownFavorites`, `PopularityFallback`, `EmptyRecommendations` (espace `MoviesAndMe`, `Alias` = `live`) |
| Version servie | Champs `revision`, `model_run_id` et `lambda_version` des enregistrements d'usage |

Sur Lambda, chaque appel à `/recommendations` écrit une ligne JSON dans les
logs. Elle contient des nombres, jamais les IDs des favoris :

```json
{"event":"recommendation","Alias":"live","lambda_version":"6","revision":"<commit>","model_run_id":"<run>","method":"als","recommendations":10,"Favorites":4,"UnknownFavorites":1,"PopularityFallback":0,"EmptyRecommendations":0}
```

Un favori est inconnu s'il n'appartient pas au catalogue d'entraînement ; un
même ID envoyé deux fois compte une fois. Pour `live`, la ligne porte aussi
les métadonnées du format EMF, et CloudWatch en tire les métriques. Les appels
à `staging`, qui viennent des vérifications de déploiement, restent dans les
logs sans créer de métriques. Hors de Lambda, rien n'est écrit.

Chaque déploiement envoie aussi deux demandes de vérification à `live` : l'une
avec un favori connu, l'autre sans favori. Les logs sont gardés 14 jours, les
métriques 15 mois.

## Tableau de bord

[movies-and-me-recommender](https://eu-north-1.console.aws.amazon.com/cloudwatch/home?region=eu-north-1#dashboards/dashboard/movies-and-me-recommender)
dans la console CloudWatch. Il montre, sur 24 heures par défaut :

- les requêtes de `live`, les erreurs 4xx et 5xx, et la latence ;
- les instances occupées, les requêtes rejetées et la mémoire ;
- les favoris par requête, la part de favoris inconnus, le repli sur la
  popularité et les réponses vides ;
- les versions qui ont servi des recommandations et les dernières erreurs de
  l'application, lues dans les logs ;
- l'état des alarmes.

Sa définition se trouve dans [monitoring/dashboard.json](../monitoring/dashboard.json).

## Alertes

Une alarme envoie un e-mail par le sujet SNS `movies-and-me-alerts`. Une
période sans requête ne déclenche rien.

| Alarme | Condition |
|---|---|
| `movies-and-me-live-5xx` | Au moins une erreur 5xx sur `live` en 5 minutes |
| `movies-and-me-throttles` | Au moins 5 requêtes rejetées en 5 minutes : les deux instances étaient occupées |
| `movies-and-me-live-latency` | p95 de la latence de `live` au-dessus de 2 s pendant 15 minutes |
| `movies-and-me-live-traffic` | Plus de 3 000 requêtes sur `live` en 5 minutes (10 par seconde), ce qui fait monter le coût |
| `movies-and-me-memory` | Une instance a utilisé plus de 800 Mo sur 1 024 |
| `movies-and-me-unknown-favorites` | Sur 15 minutes et au moins 20 favoris reçus, plus de la moitié sont inconnus |

Un démarrage à froid prend environ 3 secondes ; l'alarme de latence ne retient
donc qu'un ralentissement qui dure.

## Comparaison avec une période de référence

Le workflow **Input comparison** compare chaque jour, à 06 h 17 UTC, les
dernières 24 heures aux 7 jours précédents : nombre de requêtes, favoris par
requête, part de favoris inconnus, repli sur la popularité et réponses vides.
Il échoue si une part varie de plus de 10 points (5 pour les réponses vides) ou
si la taille moyenne des profils change de plus de moitié. GitHub envoie alors
un e-mail. Chaque période doit compter au moins 30 requêtes, sinon la
comparaison est sautée.

Le workflow peut aussi être lancé depuis l'onglet **Actions** avec d'autres
durées. Depuis un terminal disposant d'un accès AWS :

```bash
python -m pip install -r requirements-monitoring.txt
export AWS_PROFILE=movies-and-me
python scripts/compare_inputs.py --current-hours 24 --reference-hours 168
```

GitHub suspend les workflows planifiés après 60 jours sans activité dans le
dépôt ; il suffit alors de les réactiver dans l'onglet **Actions**.

## Mise en place

Avec un accès AWS administrateur, ce script crée ou met à jour le filtre de
mémoire, le sujet SNS, les alarmes et le tableau de bord :

```bash
export AWS_PROFILE=movies-and-me AWS_REGION=eu-north-1
scripts/setup_monitoring.sh adresse@example.com
```

Au premier lancement, SNS envoie un lien de confirmation à cette adresse. Les
alertes ne partent qu'une fois l'abonnement confirmé.

Le workflow de comparaison utilise le rôle `movies-and-me-recommender-monitoring`,
qui peut seulement lire les métriques (`cloudwatch:GetMetricData`). L'environnement
GitHub `monitoring`, limité à `main`, contient son ARN dans le secret
`AWS_MONITORING_ROLE_ARN`. La politique de confiance reprend celle du rôle de
déploiement, avec le sujet `...:environment:monitoring`.

## Tester les alertes

Pour vérifier l'envoi d'un e-mail sans provoquer de panne :

```bash
aws cloudwatch set-alarm-state --alarm-name movies-and-me-live-5xx \
  --state-value ALARM --state-reason "Test de notification"
```

L'alarme revient d'elle-même à l'état normal à l'évaluation suivante.

Pour provoquer une hausse de favoris inconnus, envoyer des IDs absents du
catalogue :

```bash
API=https://7xmkh4737vk2yl5ribrhs5tzei0azgje.lambda-url.eu-north-1.on.aws
for i in $(seq 1 30); do
  curl -s -o /dev/null -X POST "$API/recommendations" \
    -H "Content-Type: application/json" \
    -d "{\"favorite_movie_ids\":[$((990000000 + i)),$((991000000 + i))]}"
done
```

La part de favoris inconnus apparaît dans le tableau de bord après quelques
minutes, et l'alarme `movies-and-me-unknown-favorites` passe en alerte à la
fin de la période de 15 minutes.
