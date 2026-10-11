# Comparaison de la popularité et des genres

Les modèles de popularité et de genres ont été évalués sur la validation
MovieLens, avec les dates de `config/evaluation.json`. Une note d'au moins
4/5 sert de favori simulé. L'application, elle, transmet seulement ses
favoris TMDB.

## Protocole

Le catalogue contient les 5 223 films TMDB ayant au moins un avis positif
avant la coupure d'apprentissage. Les deux modèles reçoivent les mêmes
favoris positifs de cette période et excluent uniquement ces favoris.
Les notes plus faibles ne participent ni au profil ni aux exclusions.

La popularité compte les utilisateurs positifs distincts de l'apprentissage.
Le TF-IDF des genres utilise seulement les métadonnées canoniques du même
catalogue, avec IMAX écarté. Les ex æquo suivent les règles existantes :
ID TMDB croissant pour la popularité, ID MovieLens canonique croissant
pour les genres. Aucun modèle n'utilise les réponses de validation pour
construire son classement. Aucun réglage n'a été choisi à partir des scores.

Un utilisateur entre dans la comparaison s'il a des favoris d'apprentissage
et au moins une cible positive de validation dans le catalogue. Ce groupe
est fixé pour les deux modèles. Un profil sans genres exploitables reste
évalué avec une liste vide et un score nul. La popularité ne complète pas
le classement par genres.

Recall@10 est le nombre de cibles retrouvées dans les dix premiers résultats,
divisé par le nombre de cibles attendues. NDCG@10 utilise des gains binaires,
pondérés par `1 / log2(rang + 1)`, puis divisés par le gain du classement
idéal de `min(10, nombre de cibles)` films. Les moyennes donnent le même
poids à chaque utilisateur. Renvoyer moins de dix films ne réduit pas le
nombre de cibles attendu.

## Résultats

Le calcul principal porte sur les cibles présentes dans le catalogue.

| Modèle | Utilisateurs | Cibles connues | Recall@10 moyen | NDCG@10 moyen |
|---|---:|---:|---:|---:|
| Popularité | 20 | 333 | 4,57 % | 0,03894 |
| Genres | 20 | 333 | 0 % | 0 |

La popularité retrouve cinq cibles chez quatre utilisateurs. Ce total de
cinq sur 333 n'est pas le Recall moyen du tableau : les utilisateurs n'ont
pas tous le même nombre de cibles. Les genres renvoient dix films pour
chacun des vingt utilisateurs, mais aucun ne correspond à une cible.

Pour les mêmes utilisateurs, compter aussi les 77 cibles hors catalogue
donne 410 cibles attendues. Le Recall@10 de la popularité devient 4,16 %
et son NDCG@10 0,03481. Les deux métriques des genres restent à zéro.
Ce second calcul conserve le même groupe d'utilisateurs ; il ne mesure
pas les performances sur l'ensemble de la validation.

## Limites

La validation contient 62 utilisateurs et 5 197 avis positifs. Seuls vingt
utilisateurs ont à la fois des favoris antérieurs et une cible disponible.
Un utilisateur supplémentaire a des favoris antérieurs, mais uniquement
une cible hors catalogue. Les autres n'ont pas de favoris d'apprentissage.

Les 333 cibles du calcul principal représentent 6,4 % des avis positifs
de validation. Le catalogue couvre 4 595 cibles sur 5 197, soit 88,4 %, mais
la présence du film ne suffit pas à rendre son utilisateur éligible.
Les 602 cibles hors catalogue restent comptées dans le rapport.

La popularité fait mieux ici. Ce groupe est trop petit et trop restreint
pour en déduire la qualité pour tous les utilisateurs de Movies-And-Me.
Les demandes sans favori connu et les films inconnus devront être évalués
séparément. Le résultat nul des genres concerne ce classement et ce
protocole précis. Le test final n'a servi ni à entraîner ni à choisir
ni à mesurer les modèles.

## Reproduire le calcul

Depuis la racine du dépôt, sous PowerShell :

```powershell
.\.venv\Scripts\python.exe -m src.evaluation.compare
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Le [rapport JSON](baseline_comparison.json) conserve les métriques exactes,
les recommandations et les comptes par utilisateur, les règles, les dates,
le seuil, les versions de Python et des bibliothèques, et les empreintes
SHA-256 de `ratings.csv`, `links.csv` et `movies.csv`.
La [comparaison ALS](als_comparison.md) reprend le même protocole et les
mêmes références.
