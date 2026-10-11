# Comparaison ALS, popularité et genres

ALS est fourni par `implicit==0.7.3`, sur CPU. Les trois configurations de
[config/als.json](../config/als.json) ont été entraînées sur le même groupe
d'apprentissage, puis évaluées avec le protocole de
[comparaison des références](baseline_comparison.md). Les notes MovieLens
d'au moins 4/5
simulent des favoris ; l'application transmet seulement ses favoris TMDB.

## Conditions de comparaison

La matrice d'ALS contient 521 utilisateurs MovieLens × 5 223 films et
39 251 paires positives. Chaque paire vaut 1, indépendamment de sa note.
`implicit` applique `alpha` une fois à l'entraînement et une fois au recalcul
du profil reçu. Les notes faibles ne construisent pas le profil et ne
retirent aucun film des recommandations.

Le catalogue, les utilisateurs évalués et les favoris fournis sont communs
aux trois modèles. Le classement utilise seulement l'apprentissage : la
validation sert à mesurer les essais et à sélectionner les paramètres ALS.
Le test final n'intervient ni dans l'entraînement ni dans ce choix.

Chaque liste contient dix films, sans les favoris transmis. Les métriques
principales portent sur 20 utilisateurs et 333 cibles dans le catalogue.
Recall@10 mesure la part des cibles retrouvées ; NDCG@10 prend aussi en compte
leur position. Les moyennes donnent le même poids à chaque utilisateur.

## Résultats des essais

| Modèle ou essai | Facteurs | Régularisation | Recall@10 moyen | NDCG@10 moyen |
|---|---:|---:|---:|---:|
| Popularité | — | — | 4,57 % | 0,03894 |
| Genres | — | — | 0 % | 0 |
| `als_32` | 32 | 0,1 | 5,69 % | 0,04708 |
| `als_64` (retenu) | 64 | 0,1 | 4,81 % | 0,06231 |
| `als_32_regularized` | 32 | 1 | 5,36 % | 0,04429 |

Les essais utilisent tous `alpha=20`, `iterations=30`, `random_state=42`
et `num_threads=1`. Le critère déclaré choisit le meilleur NDCG@10, puis
le Recall@10 en cas d'égalité ; une égalité complète conserve l'ordre de
déclaration. Il retient `als_64`. `als_32` retrouve davantage de cibles en
moyenne, avec un NDCG inférieur : aucun essai ne maximise les deux métriques.

L'essai retenu retrouve sept cibles chez cinq utilisateurs, contre cinq
cibles chez quatre utilisateurs pour la popularité. Ces totaux ne sont
pas les Recall moyens du tableau : les nombres de cibles varient selon
les utilisateurs. Les genres ne retrouvent aucune cible.

## Cibles indisponibles et limites

Les vingt utilisateurs ont aussi 77 cibles hors catalogue, soit 410
cibles au total. En les incluant dans le calcul :

| Modèle | Recall@10 moyen | NDCG@10 moyen |
|---|---:|---:|
| Popularité | 4,16 % | 0,03481 |
| Genres | 0 % | 0 |
| ALS retenu | 3,82 % | 0,05612 |

ALS conserve un NDCG plus élevé, mais son Recall devient inférieur à celui
de la popularité. Le résultat dépend donc de la métrique et des cibles
comptées. Ce second tableau porte toujours sur les mêmes vingt utilisateurs.

L'ensemble de la validation contient 62 utilisateurs et 5 197 avis
positifs. Les 333 cibles du calcul principal en représentent seulement
6,4 %. Le catalogue couvre 4 595 avis positifs (88,4 %) ; 602 restent
hors catalogue. Un film connu ne suffit pas à rendre son utilisateur
évaluable : il faut aussi des favoris dans l'apprentissage.

Ce groupe limité ne permet pas de conclure sur tous les utilisateurs de
Movies-And-Me. Tous les profils évalués contiennent des favoris connus ;
la qualité du secours par popularité n'est donc pas mesurée ici. Les notes
MovieLens restent une approximation des favoris réels de l'application.

## Modèle sauvegardé et reproduction

Le modèle retenu reste entraîné sur l'apprentissage uniquement. Ses facteurs,
sa matrice, ses correspondances d'IDs, la popularité et son manifeste sont
dans `artifacts/implicit-als/`, ignoré par Git. Le chargement vérifie les
dimensions et les empreintes, puis permet de recommander sans relire les CSV.
Le réentraînement sur apprentissage et validation, puis la mesure finale
sur le test, restent à faire après fixation des choix.

Depuis la racine du dépôt, sous PowerShell :

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.training.train
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

La commande refuse d'écraser un dossier de modèle existant. Pour relancer
l'expérience, choisir des chemins libres, par exemple :

```powershell
.\.venv\Scripts\python.exe -m src.training.train --artifact-dir artifacts/implicit-als-run-2 --output reports/als_comparison-run-2.json
```

Le [rapport JSON](als_comparison.json) contient les valeurs exactes, les
recommandations par utilisateur, les paramètres, les dates, les versions
des bibliothèques et les empreintes des sources. Le
[notebook exécuté](../notebooks/03_implicit_als.ipynb) montre les résultats,
deux profils TMDB, les favoris vides ou inconnus et la sauvegarde/relecture.
Le [README](../README.md#suivi-et-registre-mlflow) décrit le suivi MLflow
des entraînements et l'enregistrement du modèle.
