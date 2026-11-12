# Roadmap — worldmodel

## Plan d'action

Objectif final : un process local, sur RTX 4060 8 Go / 48 Go RAM / Windows, qui permet de lancer et de piloter la génération d'un world model interactif depuis une UI web lancée par `run.py` (autoreload).

1. Préparer un environnement isolé sur D: (Python, CUDA, cache des poids).
2. Construire un banc de test commun : une interface d'adaptateur par modèle et des scénarios verrouillés.
3. Intégrer les candidats un par un et les mesurer.
4. Choisir le modèle à partir des mesures, pas de la réputation.
5. Construire l'UI de pilotage autour du modèle retenu, en gardant l'interface d'adaptateur pour pouvoir changer de modèle.
6. Finaliser le process : enregistrement, export, documentation de lancement.

### Candidats issus de la recherche web (2026-11-10)

Aucun world model interactif open source n'est documenté comme tournant sur 8 Go de VRAM. Les chiffres ci-dessous sont des hypothèses à mesurer en phase 3.

| Candidat | Taille | Référence matérielle publiée | Pronostic 8 Go |
|---|---|---|---|
| DIAMOND Atari | ~4 M | aucune exigence publiée | très probable |
| DIAMOND CS:GO (branche `csgo`) | ~381 M | ~10 FPS sur RTX 3090 | probable, FPS réduit |
| Open-Oasis 500M (Minecraft) | 500 M | 12 Go annoncés par un site tiers non officiel | incertain |
| Matrix-Game 2.0 (Skywork) | 1,8 B | 25 FPS sur H100, dépend d'Apex et FlashAttention | incertain, offload ou quantification nécessaire, difficile sous Windows |
| Yume 1.5 / Hunyuan-GameCraft | 5 B et plus | base HunyuanVideo à 48-80 Go | écartés sauf découverte contraire |

### Risques identifiés

- Python 3.13 installé : la plupart des dépôts ciblent Python 3.10. Prévoir un Python 3.10 dédié sur D: (via `uv`).
- Apex et FlashAttention se compilent mal sous Windows. Matrix-Game 2.0 peut exiger WSL2.
- Autoreload et modèle chargé : un rechargement du serveur rechargerait les poids, ce qui prend du temps et fragmente la VRAM. L'architecture sépare le processus UI (rechargé) du worker d'inférence (persistant).

### Sources
- https://github.com/eloialonso/diamond
- https://arxiv.org/pdf/2405.12399
- https://the-decoder.com/ai-model-simulates-counter-strike-with-10-fps-on-a-single-rtx-3090/
- https://github.com/etched-ai/open-oasis
- https://huggingface.co/Etched/oasis-500m
- https://oasis-ai.org/documentation.php
- https://huggingface.co/Skywork/Matrix-Game-2.0
- https://arxiv.org/pdf/2508.13009
- https://app.cinevva.com/guides/frontier-gen-ai-models
- https://arxiv.org/pdf/2511.23429

---

## Phase 1 — Environnement local [FAIT]

- Python 3.10 dédié, venv dans `D:\ServOMorph\WorldModel\.venv`.
- PyTorch CUDA compatible avec le pilote 610.88.
- Cache Hugging Face et poids sur D: (`HF_HOME`, dossier `models/` ignoré par git).
- Script `tools/check_env.py` : GPU visible, VRAM, version CUDA, test d'allocation.
- Tests : `check_env.py` passe, test pytest de détection du GPU.

**⏸ Checkpoint** — Demander à l'utilisateur de faire `/compact` avant de continuer.
Attendre sa réponse écrite. Ne pas commencer la phase suivante sans confirmation.

## Phase 2 — Banc de test [FAIT]

- Interface `WorldModelAdapter` : `load()`, `reset(seed, image|prompt)`, `step(action) -> frame`, `unload()`.
- Adaptateur factice (bruit déterministe) pour tester le banc sans GPU.
- Scénarios verrouillés : N séquences d'actions fixes et seeds fixes par famille de jeu.
- Métriques : VRAM pic, RAM pic, temps de chargement, FPS moyen et p95, dérive visuelle sur la durée (similarité image début/fin), plantage ou OOM.
- Sortie : vidéo mp4 par run et `bench/results.json`, plus une grille de notation manuelle de la qualité.
- Tests : pytest sur le banc avec l'adaptateur factice.

**⏸ Checkpoint** — Demander à l'utilisateur de faire `/compact` avant de continuer.
Attendre sa réponse écrite. Ne pas commencer la phase suivante sans confirmation.

## Phase 3 — Intégration et mesure des candidats [EN COURS]

Ordre : du plus probable au moins probable sur 8 Go.
1. DIAMOND Atari.
2. DIAMOND CS:GO.
3. Open-Oasis 500M.
4. Matrix-Game 2.0 : fp16/bf16, offload CPU, éventuellement WSL2. Abandon documenté si OOM ou dépendances bloquantes.

- Un adaptateur par candidat, chaque environnement isolé si les dépendances divergent.
- Exécution du banc complet pour chacun.
- Tests : chaque adaptateur passe le scénario de fumée (10 frames sans erreur).

**⏸ Checkpoint** — Demander à l'utilisateur de faire `/compact` avant de continuer.
Attendre sa réponse écrite. Ne pas commencer la phase suivante sans confirmation.

## Phase 4 — Sélection [TODO]

- Rapport `bench/rapport.md` : tableau comparatif, vidéos, limites.
- Critères pondérés (à valider avec l'utilisateur) : interactivité (FPS), stabilité longue durée, qualité visuelle, liberté de contenu (prompt ou image initiale), coût VRAM.
- Décision consignée dans `_contexte/contexte.md` par `/close`.

**⏸ Checkpoint** — Demander à l'utilisateur de faire `/compact` avant de continuer.
Attendre sa réponse écrite. Ne pas commencer la phase suivante sans confirmation.

## Phase 5 — UI de pilotage [TODO]

- `run.py` à la racine : lance le serveur web en autoreload et le worker d'inférence persistant hors du rechargement.
- Backend FastAPI + WebSocket ; frontend HTML/JS (canvas, capture clavier et souris).
- Fonctions : choix du modèle (adaptateurs disponibles), image ou prompt initial, seed, démarrer, pause, réinitialiser, paramètres (résolution, pas de diffusion), affichage FPS et VRAM, enregistrement de la session.
- Tests : pytest sur l'API avec l'adaptateur factice ; test manuel de jeu ajouté à `tests_manuels.md`.

**⏸ Checkpoint** — Demander à l'utilisateur de faire `/compact` avant de continuer.
Attendre sa réponse écrite. Ne pas commencer la phase suivante sans confirmation.

## Phase 6 — Finalisation du process [TODO]

- Export mp4 des sessions, historique des sessions dans l'UI.
- Lancement du banc depuis l'UI pour réévaluer un nouveau modèle.
- `README.md` : installation et lancement.
- Tests : parcours complet installation, banc, UI sur machine propre (venv recréé).

**⏸ Checkpoint** — Demander à l'utilisateur de faire `/compact` avant de continuer.
Attendre sa réponse écrite. Ne pas commencer la phase suivante sans confirmation.
