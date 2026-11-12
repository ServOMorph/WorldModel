# Contexte — worldmodel

## Objectif (immuable sauf décision explicite)
Installer et utiliser en local un world model interactif sur une RTX 4060 8 Go.

## Stack / contraintes techniques (stable, rarement modifié)
Windows, Python, nvidia-smi

## État actuel (réécrit intégralement à chaque /close)
Phases 1 et 2 faites (environnement, banc de test). Phase 3 en cours : DIAMOND Atari, DIAMOND CS:GO et Open-Oasis mesurés ; Matrix-Game 2.0 téléchargé, essai en attente du signal de l'utilisateur.
Aucune utilisation du GPU sans ce signal.

## Décisions structurantes (append only — 10 entrées max, 5 lignes max/entrée, archiver au-delà)
- 2026-11-09 : Initialisation du protocole vibecoding.
- 2026-11-12 : Code externe cloné dans `third_party/`, poids dans `models/` et `D:\HuggingFaceCache`, tous deux ignorés par git. Open-Oasis via le dépôt tiers `cocktailpeanut/open-oasis` (officiel introuvable).
