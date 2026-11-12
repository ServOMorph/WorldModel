# Signals — worldmodel   (MAJ 2026-11-12)

## Actions ouvertes
<!-- ≤ 20 entrées (ligne « Backlog » exclue) ; dette et findings non urgents -> _contexte/backlog.md -->
- [P1|attente] Tenter Matrix-Game 2.0 (candidat 4) sur le GPU, après signal de l'utilisateur (interdiction d'utiliser la VRAM sans son signal)
  - fait quand: adaptateur `worldmodel/adapters/matrix_game.py` mesuré sur smoke/short/long, ou abandon documenté (OOM, dépendance Windows bloquante).
  - réf: `models/matrix-game-2/`, `third_party/matrix-game/Matrix-Game-2/inference_streaming.py`.
- [P2|ouvert] Noter manuellement les vidéos de `bench/runs/` dans `bench/grille_notation.md` (Phase 4)
  - fait quand: grille remplie pour les 4 modèles mesurés.
  - réf: `bench/grille_notation.md`, `bench/runs/`.
- [P2|ouvert] Compléter `tests/test_adapters.py` si Matrix-Game est intégré ; le test smoke Open-Oasis est écrit mais non exécuté via pytest
  - fait quand: `pytest tests -q` passe avec les 4 adaptateurs.
  - réf: `tests/test_adapters.py`.
- [P2|ouvert] Mesure VRAM globale à la carte (bureau Windows inclus) : à corriger ou à documenter dans le rapport Phase 4
  - fait quand: `bench/metrics.py` mesure la VRAM du processus (torch.cuda.max_memory_allocated) ou la limite est notée dans `bench/rapport.md`.
  - réf: `bench/metrics.py`.

## Dernière session (2026-11-12)
# Session du 2026-11-12

## Décisions prises
- Phase 3 démarrée : un adaptateur par candidat, code externe cloné dans `third_party/` (ignoré par git), poids dans `models/` et `D:\HuggingFaceCache`.
- Open-Oasis : code tiers `cocktailpeanut/open-oasis` retenu faute de dépôt officiel accessible, lu avant exécution.
- Matrix-Game 2.0 : poids téléchargés (11 Go), essai différé jusqu'au signal de l'utilisateur.

## Livrables produits ou modifiés
- `worldmodel/adapters/diamond_atari.py`, `diamond_csgo.py`, `open_oasis.py` : créés, mesurés.
- `tests/test_adapters.py` : 3 tests smoke ; Atari et CS:GO passés, Open-Oasis non lancé via pytest.
- `bench/results.json`, `bench/runs/` : résultats des 3 modèles sur smoke/short/long.
- `roadmap_worldmodel.md` : statuts Phases 1-3 mis à jour.

## Hypothèses validées / invalidées
- VALIDE : DIAMOND Atari (11 FPS, 1,2 Go VRAM, dérive 0,02), CS:GO (4,6 FPS moyen, p95 0,49, VRAM ~8 Go, dérive 0,67), Open-Oasis (0,33 FPS, 7,3 Go VRAM, dérive 0,78) tournent sur la 4060.
- EN ATTENTE : Matrix-Game 2.0 (officiel : 24 Go VRAM, Linux, FlashAttention).

## Prochaine étape exacte
Sur signal de l'utilisateur, écrire l'adaptateur Matrix-Game (offload CPU, repli sans FlashAttention) et le mesurer, sinon l'abandonner ; puis checkpoint `/compact` de la Phase 3.

## Question bloquante pour la session suivante
Aucune
