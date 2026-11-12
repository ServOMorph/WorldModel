# WorldModel

Génération de world models interactifs en local (RTX 4060 8 Go, Windows).

## État actuel
Phases 1 (environnement) et 2 (banc de test) faites. Phase 3 en cours : DIAMOND Atari, DIAMOND CS:GO et Open-Oasis mesurés, Matrix-Game 2.0 en attente. UI de pilotage (`run.py`) prévue en Phase 5.

## Structure
- `worldmodel/adapters/` : interface `WorldModelAdapter` et un adaptateur par candidat.
- `bench/` : scénarios, métriques, runner, résultats (`results.json`, `runs/`).
- `tests/` : tests de l'environnement, du banc et des adaptateurs.
- `third_party/`, `models/` : code externe et poids, ignorés par git.
- `roadmap_worldmodel.md` : plan par phases.
