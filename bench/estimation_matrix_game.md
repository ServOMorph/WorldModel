# Estimation mémoire — Matrix-Game 2.0 (offload CPU, 8 Go VRAM / 48 Go RAM)

Estimation sans chargement GPU, à partir des fichiers de poids et du code de `third_party/matrix-game/Matrix-Game-2/`. Aucune mesure réelle.

## Faits lus dans le code et les fichiers

- DiT (`base_distilled_model/base_distill.safetensors`) : 1 619 242 304 paramètres, stockés en F32 (6,48 Go sur disque).
- Config : 30 couches, dim 1536, `local_attn_size` 6 (configs distilled), 880 tokens par frame latente.
- VAE Wan2.1 (`Wan2.1_VAE.pth`) : 0,51 Go sur disque.
- CLIP `open-clip-xlm-roberta-large-vit-huge-14.pth` : 4,77 Go sur disque, utilisé par `vae.clip.encode_video` lors du reset.
- `InteractiveGameInference._init_models` : `pipeline.to(cuda, bf16)`, décodeur VAE en fp16 compilé (`torch.compile`, `max-autotune-no-cudagraphs`), `vae.to(cuda, bf16)`. `WanxVAEWrapper.to` déplace aussi le CLIP sur GPU.
- Aucun code d'offload CPU dans `inference_streaming.py` (`grep offload` : seul `wan/image2video.py` en a un, non utilisé par ce pipeline).
- Attention : `flash_attn` absent, repli sur `scaled_dot_product_attention` (`wan/modules/attention.py`).

## Estimation VRAM pic (hypothèses indiquées)

| Composant | Estimation | Base |
|---|---|---|
| DiT en bf16 | 3,24 Go | 1,62 G params x 2 o (fait) |
| VAE complet + décodeur fp16 | ~0,35 Go | hypothèse : ~127 M params |
| CLIP en bf16 | ~2,4 Go | hypothèse : 4,77 Go F32, soit ~1,19 G params |
| KV cache | ~1,0 Go | 6 x 880 tokens x 30 couches x (k,v) x 1536 x 2 o, hors caches souris/clavier |
| Activations et tiling VAE | 0,5 à 1 Go | hypothèse, non mesuré |
| Contexte CUDA et allocateur | ~0,5 Go | hypothèse |
| Bureau Windows | inconnu | non mesuré |

- Sans offload du CLIP : ~8,2 Go hors bureau. Dépasse 8 Go, et dépasse le seuil de 7,5 Go de VertIA_v3. Non faisable en l'état.
- Avec CLIP déplacé sur CPU après `encode_video` (et VAE encoder après le premier encode) : ~5,8 Go hors bureau, soit environ 6,5 à 7 Go avec le bureau. Sous 7,5 Go, sous condition.

## Estimation RAM (CPU)

- Chargement des poids en F32 : 6,48 + 0,51 + 4,77 = ~11,8 Go.
- Pic RAM estimé 12 à 24 Go selon les copies au chargement (hypothèse). Compatible avec 48 Go.

## Faisabilité

- **En l'état : non.** Le script charge tout sur GPU, et le pic estimé (~8,2 Go) dépasse 8 Go.
- **Sous conditions : oui, à mesurer.** Il faut modifier le code (offload CLIP et VAE encoder, éventuellement désactiver `torch.compile`). Ce n'est pas dans le dépôt tiers tel qu'il est.

## Risques non résolus

- `torch.compile` en mode `max-autotune` sous Windows : dépendance Triton, risque bloquant non testé.
- Le projet cible 24 Go (A100/H100) ; aucun résultat publié sur 8 Go.
- Les chiffres ci-dessus sont des estimations : seule une mesure par processus (`torch.cuda.max_memory_allocated`) les confirmera, et ce n'est possible qu'après accord GPU (jeton VertIA_v3).

## Prochaine étape

Écrire l'adaptateur avec offload et mesurer uniquement après jeton VertIA_v3 et estimation validée.
