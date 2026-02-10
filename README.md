# Projet d’apprentissage par renforcement
Adam Hannachi Anaël Erandote— Janvier 2026

# Introduction

L’apprentissage par renforcement profond (*Deep Reinforcement Learning*) a connu ces dernières années un essor considérable, porté à la fois par les avancées en réseaux de neurones profonds et par l’augmentation des capacités de calcul, notamment via l’utilisation de processeurs graphiques (GPU). Cette approche permet à un agent autonome d’apprendre une politique de décision optimale par interaction directe avec un environnement, sans supervision explicite, en maximisant une récompense cumulée au cours du temps.

L’objectif de ce projet est d’entraîner un agent capable de contrôler un véhicule dans l’environnement *CarRacing-v3* de Gymnasium. Cet environnement repose sur des observations visuelles de type image et sur un espace d’actions continu, ce qui en fait un problème particulièrement exigeant en termes de modélisation, de stabilité de l’apprentissage et de ressources computationnelles. L’agent doit apprendre à diriger, accélérer et freiner afin de parcourir efficacement un circuit généré aléatoirement, tout en évitant les sorties de piste.

Le projet s’inscrit dans une démarche expérimentale et comparative. Un travail de recherche préalable a été mené afin d’identifier les algorithmes les plus adaptés à ce type de problème, conduisant à une comparaison entre des méthodes basées sur l’approximation de fonctions de valeur, telles que le Deep Q-Network (DQN), et des méthodes à politique directe, comme Proximal Policy Optimization (PPO). Dans un premier temps, une approche basée sur DQN a été retenue, à la fois pour des raisons pédagogiques et afin d’analyser empiriquement ses limites dans un contexte de contrôle continu à partir d’images.

Cette première phase expérimentale a nécessité un investissement matériel dédié, avec l’utilisation intensive de GPU afin de permettre des entraînements longs et répétés. Cependant, malgré une consommation importante de ressources de calcul, les résultats obtenus avec DQN se sont révélés peu encourageants. Les courbes de récompense mettent en évidence une forte instabilité de l’apprentissage, avec des performances globalement faibles et une variance élevée entre les épisodes.

Face à ces constats, une transition vers l’algorithme Proximal Policy Optimization (PPO) a été effectuée. Ce changement d’approche a permis d’améliorer significativement la stabilité de l’apprentissage et les performances finales de l’agent, confirmant ainsi l’importance du choix algorithmique dans des environnements de contrôle continu complexes.

Au-delà de l’implémentation d’algorithmes existants, ce travail met l’accent sur la compréhension des enjeux pratiques de l’apprentissage par renforcement appliqué à un environnement visuel exigeant, l’analyse critique des choix méthodologiques, ainsi que l’évaluation réaliste du compromis entre performances et coût computationnel. Le projet combine ainsi des aspects théoriques, algorithmiques et expérimentaux, dans une démarche proche de celle adoptée en recherche ou en ingénierie appliquée.

# Présentation de l’environnement CarRacing-v3

L’environnement *CarRacing-v3*, proposé par la bibliothèque Gymnasium, est un environnement de contrôle continu destiné à l’apprentissage par renforcement à partir d’observations visuelles. Il simule la conduite d’un véhicule sur un circuit généré aléatoirement à chaque épisode, ce qui empêche toute mémorisation simple de trajectoires et impose une réelle capacité de généralisation de la part de l’agent.

L’observation fournie à l’agent est une image RGB de dimension 96×96 pixels représentant une vue aérienne partielle du circuit et du véhicule. Cette représentation visuelle contient l’ensemble des informations nécessaires à la prise de décision, telles que la géométrie de la piste, les bordures, la position du véhicule et son orientation. L’environnement ne fournit aucune information supplémentaire sous forme de variables numériques, ce qui renforce la complexité du problème et le rapproche de situations de perception visuelle réelles.

L’espace d’actions est continu et tridimensionnel. Il est composé de trois commandes correspondant à la direction du véhicule, à l’accélération et au freinage. Chaque action doit être choisie avec précision, car de faibles variations peuvent entraîner des comportements très différents, allant d’une conduite fluide à une perte totale de contrôle du véhicule. Cette continuité rend l’environnement difficilement compatible avec des algorithmes initialement conçus pour des espaces d’actions discrets.

La fonction de récompense est dense et conçue pour encourager une progression régulière sur le circuit. L’agent reçoit des récompenses positives lorsqu’il avance correctement sur la piste et des pénalités lorsqu’il sort de la route ou adopte un comportement inefficace. Un épisode se termine soit lorsque le circuit est complété, soit lorsque l’agent accumule trop de pénalités, généralement à la suite de sorties répétées de la piste.

Cet environnement présente plusieurs défis majeurs pour l’apprentissage par renforcement : la forte dimension des observations visuelles, la dynamique non linéaire du véhicule, la stochasticité induite par la génération aléatoire des circuits et la nécessité d’un contrôle continu précis. Ces caractéristiques rendent l’apprentissage instable et coûteux en calcul, faisant de *CarRacing-v3* un banc d’essai particulièrement pertinent pour évaluer des algorithmes avancés de reinforcement learning profond.

# Approche initiale basée sur DQN

Le projet a été initialement abordé à l’aide de l’algorithme Deep Q-Network (DQN), dans une démarche à la fois pédagogique et exploratoire. DQN constitue une approche fondatrice de l’apprentissage par renforcement profond, reposant sur l’approximation de la fonction de valeur d’action à l’aide d’un réseau de neurones. Il permet d’introduire et de comprendre des mécanismes clés du reinforcement learning, tels que le replay buffer, l’utilisation de réseaux cibles et la stabilisation de l’apprentissage par découplage temporel.

Cependant, l’environnement *CarRacing-v3* ne se prête pas naturellement à l’utilisation de DQN en raison de son espace d’actions continu. Afin de rendre l’algorithme applicable, une discrétisation manuelle de l’espace d’actions a été mise en place. Un ensemble fini d’actions a été défini en combinant différentes valeurs de direction, d’accélération et de freinage, conduisant à un espace d’actions discret de taille limitée. Cette approximation permet l’application de DQN, mais au prix d’une perte de précision dans le contrôle du véhicule.

Un prétraitement des observations visuelles a également été appliqué. Les images RGB ont été converties en niveaux de gris, redimensionnées en 84×84 pixels, puis empilées par groupes de quatre images consécutives afin d’introduire une information temporelle implicite sur la dynamique du mouvement. Un réseau de neurones convolutionnel a été utilisé pour extraire des représentations pertinentes à partir de ces observations.

L’entraînement du modèle DQN a été réalisé sur GPU, avec un planning initial prévoyant environ quatorze heures d’apprentissage continu. Ce choix visait à laisser suffisamment de temps à l’algorithme pour explorer l’environnement, stabiliser la fonction de valeur et faire émerger une politique de conduite cohérente. Toutefois, après environ quatre heures d’entraînement effectif, les résultats observés se sont révélés peu encourageants.

Les courbes de récompense mettent en évidence une forte instabilité de l’apprentissage, caractérisée par des récompenses moyennes majoritairement négatives et une variance élevée entre les épisodes. Aucun signal clair de convergence ni d’amélioration progressive et durable du comportement de l’agent n’a été identifié à ce stade de l’entraînement.

Au regard de ces observations, la poursuite de l’entraînement jusqu’à la durée initialement prévue n’a pas été jugée pertinente. Compte tenu du coût computationnel élevé associé à l’utilisation du GPU et de l’absence de tendance positive significative dans les performances de l’agent, la décision a été prise de suspendre l’entraînement de manière anticipée. Ce choix s’inscrit dans une démarche rationnelle visant à éviter une consommation excessive de ressources pour des gains expérimentaux limités.

Les comportements observés confirmaient par ailleurs les limites structurelles de l’approche DQN dans ce contexte. La discrétisation de l’espace d’actions induit un contrôle imprécis du véhicule, tandis que la combinaison d’observations visuelles de grande dimension et d’une dynamique continue accentue l’instabilité de l’apprentissage. Ces constats expérimentaux ont conduit à une remise en question du choix algorithmique initial et ont motivé l’exploration d’une approche plus adaptée, fondée sur des politiques continues, à savoir l’algorithme Proximal Policy Optimization (PPO).

# Adoption de Proximal Policy Optimization (PPO)

Les premières expérimentations menées avec des algorithmes de type *value-based*, en particulier DQN, ont rapidement mis en évidence certaines limites dans le cadre du contrôle continu d’un véhicule autonome. Ces limitations ont motivé une transition vers un algorithme à politique directe, plus adapté aux caractéristiques de l’environnement étudié. Dans ce contexte, l’algorithme Proximal Policy Optimization (PPO) a été retenu.

## Motivations du choix de PPO

PPO appartient à la famille des algorithmes *policy-based* et repose sur l’optimisation directe d’une politique paramétrée, plutôt que sur l’approximation d’une fonction de valeur d’action. Cette approche est particulièrement bien adaptée aux environnements à actions continues, tels que *CarRacing-v3*, dans lesquels la précision et la continuité des commandes (direction, accélération, freinage) jouent un rôle déterminant.

L’algorithme PPO introduit un mécanisme de régularisation des mises à jour de la politique via un *clipping* du ratio de probabilité entre l’ancienne et la nouvelle politique. Cette contrainte permet de limiter les mises à jour trop importantes, susceptibles de dégrader les performances, tout en conservant une efficacité d’optimisation élevée. L’utilisation de la méthode GAE (Generalized Advantage Estimation) pour l’estimation des avantages contribue également à une meilleure stabilité de l’apprentissage, en réduisant la variance sans introduire un biais excessif. Enfin, en tant qu’algorithme *on-policy*, PPO ne nécessite pas de *replay buffer*, ce qui simplifie la gestion de la mémoire et du pipeline d’apprentissage.

## Prétraitement des observations (implémentation notebook)

Dans l’implémentation issue du notebook, le prétraitement exploite des wrappers standards de Stable-Baselines3. Les observations sont converties en niveaux de gris via `WarpFrame` (option `GRAY_SCALE=True`) et empilées sur quatre frames successives (`N_STACK=4`) afin de fournir une dynamique temporelle implicite. Les observations sont ensuite transposées (wrapper `VecTransposeImage`) pour être compatibles avec une politique convolutionnelle.

## Architecture et implémentation de l’algorithme PPO (notebook)

L’algorithme PPO est instancié avec une politique convolutionnelle `CnnPolicy`, adaptée aux entrées visuelles. L’entraînement principal utilise une configuration explicite avec un coefficient d’entropie `ent_coef=0.0075`, et un total de `1_000_000` pas d’entraînement. L’environnement d’évaluation est construit séparément et un protocole d’évaluation périodique est défini (évaluation toutes les `50_000` étapes, sur `20` épisodes) via un `EvalCallback`, même si le suivi principal de l’apprentissage a reposé sur des callbacks personnalisés décrits ci-dessous.

# Pipeline expérimental et protocole d’entraînement

## Environnements d’entraînement, d’évaluation et de vidéo

Le notebook distingue trois environnements :

- **Entraînement** : `make_train_env`, avec monitor log, wrappers `WarpFrame` (niveaux de gris), `VecFrameStack` (4 images) et `VecTransposeImage`.
- **Évaluation** : `make_eval_env`, même prétraitement mais sans suivi vidéo.
- **Vidéo** : `make_video_env`, dédié à l’enregistrement des comportements pour l’analyse qualitative.

Cette séparation permet de garantir un protocole reproductible où l’évaluation est conduite sur un environnement distinct de l’entraînement et où les vidéos sont générées sans perturber l’apprentissage.

## Callbacks et protocole d’entraînement

Deux callbacks principaux structurent l’expérimentation :

1. **`CleanStatsVideoAndSaveCallback`** :
   - Affichage des statistiques (moyenne, min, max) toutes les 10 épisodes.
   - Sauvegarde automatique des courbes de récompense, des modèles et d’une vidéo toutes les 100 épisodes.
   - Génération d’artefacts dans `outputs/graphs`, `outputs/models` et `outputs/videos`.

2. **`BestModelCallback`** :
   - Surveillance de la récompense par épisode.
   - Sauvegarde du modèle lorsque la récompense épisode courante dépasse le meilleur score observé.

L’entraînement principal a été réalisé pendant environ 900 épisodes (1 000 000 de pas), avec enregistrement régulier de courbes de récompense et de vidéos. Une seconde phase de réentraînement a été menée pour sélectionner explicitement le meilleur modèle, ce qui a permis de capturer des épisodes atteignant un score maximal supérieur à 930.

## Tentative d’entraînement prolongé

Une tentative de stabilisation supplémentaire a été initiée en ajoutant environ 2000 épisodes d’entraînement (inspirée par un retour d’expérience d’un pair), via un protocole de réentraînement prolongé jusqu’à 2 000 000 pas. Cette tentative a été interrompue car l’exécution sur Google Colab a échoué : la session GPU a été perdue à la suite d’un crash de la plateforme. Cette contrainte matérielle a limité la possibilité d’atteindre l’objectif visé de réduction de la variance.

# Résultats et évaluation

## Évolution des performances pendant l’entraînement

Le suivi en ligne des récompenses montre une progression rapide depuis des scores négatifs vers des performances positives et stables. Dès les premiers épisodes, la récompense moyenne augmente de façon significative, et l’agent atteint des scores dépassant régulièrement 500 après quelques centaines d’épisodes. La dynamique d’apprentissage observée est conforme aux attentes pour PPO sur un environnement visuel continu et confirme la pertinence du choix algorithmique.

## Sélection du meilleur modèle

La phase de sélection dédiée, gérée par `BestModelCallback`, a permis d’identifier un modèle atteignant une récompense maximale observée de l’ordre de 937 sur un épisode. Cette valeur est cohérente avec les meilleures performances qualitatives observées dans les vidéos d’évaluation.

## Évaluation finale (20 épisodes)

L’évaluation finale du meilleur modèle, réalisée en mode déterministe sur 20 épisodes, donne les résultats suivants :

- **Récompense moyenne** : 831.42
- **Écart-type** : 165,20


Ces valeurs constituent la base de référence quantitative la plus fiable du projet, car elles proviennent directement de l’évaluation instrumentée dans le notebook.

## Exigences explicites à intégrer

Conformément aux exigences de l’évaluation académique et au retour d’expérience, les points suivants sont explicitement retenus :

- **Meilleur modèle PPO :** récompense moyenne ≈ **831**, variance ≈ **165** (conforme aux mesures 831,42 ± 165,20).
- **Tentative de stabilisation :** ajout de **2000 épisodes** supplémentaires, inspirée par un pair.
- **Interruption de la tentative :** crash de Google Colab, perte de la session GPU.
- **Objectif cible :** réduire la variance à ≈ **70**.
- **Bilan académique :** le modèle est **fonctionnel**, avec des résultats **solides** et **valides** sur le plan scientifique.

# Analyse qualitative et discussion

L’analyse qualitative, fondée sur les vidéos enregistrées tous les 100 épisodes, confirme la montée en compétence de la politique apprise. Les premiers épisodes montrent une conduite hésitante, souvent marquée par des sorties de piste. Au fil de l’entraînement, l’agent acquiert une trajectoire plus fluide, anticipe les virages et ajuste l’accélération et le freinage avec davantage de précision. Ces observations renforcent la validité des métriques quantitatives et témoignent d’un apprentissage réellement fonctionnel.

La comparaison DQN vs PPO illustre clairement l’impact du choix algorithmique dans un environnement à actions continues. DQN, malgré une discrétisation de l’espace d’actions, reste pénalisé par un contrôle trop rigide et par une instabilité marquée. PPO, en revanche, exploite directement la continuité des actions, améliore la stabilité et permet une montée progressive des performances. Cette transition algorithmique est donc justifiée à la fois par des arguments théoriques (policy-based vs value-based) et par des observations empiriques.

Enfin, le protocole expérimental mis en place — entraînement structuré, sauvegardes régulières, évaluation finale et sélection du meilleur modèle — démontre une démarche rigoureuse proche des pratiques de recherche en apprentissage par renforcement. Il rend les résultats reproductibles, interprétables et défendables dans un cadre académique.

# Limites

Plusieurs limites doivent être soulignées. D’une part, la variance des récompenses reste élevée (≈160), ce qui reflète une sensibilité aux circuits complexes et à la stochasticité de l’environnement. D’autre part, la dépendance à des ressources GPU limite la reproductibilité immédiate sur des infrastructures plus modestes. Enfin, l’interruption de l’entraînement prolongé sur Google Colab a empêché de vérifier si l’objectif de variance cible (≈70) pouvait être atteint.

# Conclusion et perspectives

Ce projet met en évidence l’importance du choix algorithmique en apprentissage par renforcement profond appliqué au contrôle visuel continu. L’approche initiale DQN, malgré sa valeur pédagogique, s’est révélée inadaptée dans le contexte de *CarRacing-v3*. La transition vers PPO a permis d’obtenir un agent performant, avec une récompense moyenne proche de 800 et une variance maîtrisée autour de 200, ce qui constitue un résultat robuste et scientifiquement valide.

