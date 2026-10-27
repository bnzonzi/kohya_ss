# Compte rendu — Nemeriys SDXL v2

Run Kohya lu dans TensorBoard, publié dans le portail NEMOKID. Pas de nouvel entraînement.

## Statut

Terminé. 6000 / 6000 steps. Fin le 27 octobre 2026 à 13:31 (Europe/Paris). Aucun processus `sdxl_train_network` ni `accelerate` au moment de la lecture.

Le run NaN `nemeris_1026`, arrêté au step 189, est écarté. Le run de 1200 steps du 26 octobre (`training_nemeriys_sdxl.log`) est antérieur.

## Meilleur loss

| | |
| --- | --- |
| Step | 2100 |
| loss/average | 0.078933 |
| loss/current | 0.122103 |
| Checkpoint | `/home/nzonzi/ComfyUI/SDXL_trainer_output/nemeriys_sdxl_v2-step00002100.safetensors` |
| Norme \|\|lora_up\|\|₂ | 1.2552 |

Dernier loss/average, step 6000 : 0.087004. Checkpoint final : `/home/nzonzi/ComfyUI/SDXL_trainer_output/nemeriys_sdxl_v2.safetensors` (même horodatage que `nemeriys_sdxl_v2-step00006000.safetensors`).

Le plus haut loss/average parmi les sauvegardes est le step 5700 (0.090326). Tous les steps de `loss/average` et `loss/current` de ce run sont finis : 6000 valeurs, aucun NaN.

Deux creux ne sont pas des checkpoints. Au step 2, loss/average vaut 0.025289 : la moyenne vient de démarrer. Au step 5756, loss/current vaut 0.000977 : un seul batch. Le classement ne retient que les fichiers écrits tous les 300 steps.

## Classement des checkpoints

Sauvegardes tous les 300 steps, du meilleur loss/average au pire. Le tableau complet est en tête du portail.

1. step 2100 — 0.078933 — `nemeriys_sdxl_v2-step00002100.safetensors`
2. step 1200 — 0.079283
3. step 300 — 0.079751
4. step 1500 — 0.080212
5. step 2400 — 0.080268
6. step 3300 — 0.080422
7. step 3000 — 0.081320
8. step 4200 — 0.081488
9. step 3900 — 0.081629
10. step 1800 — 0.082512
11. step 3600 — 0.082599
12. step 5100 — 0.083292
13. step 600 — 0.084051
14. step 2700 — 0.084123
15. step 900 — 0.085292
16. step 4800 — 0.085453
17. step 4500 — 0.086039
18. step 6000 — 0.087004
19. step 5400 — 0.087739
20. step 5700 — 0.090326

## Config et dataset

- Config : `training_output/nemeris_sdxl/config_lora_nemeriys_sdxl.toml`
- Dataset : `training_output/nemeris_sdxl/dataset_nemeriys_sdxl.toml`
- Images : 82 png + 82 captions `.txt`, repeats 4, résolution 1024
- Dossier : `/home/nzonzi/ComfyUI/output/ComfyUI_Flux_Dataset_Nemeriys/dossier`
- Régularisation : `training_output/nemeris_flux/reg_260_from_1_woman` (class token `woman`)
- Base : `/home/nzonzi/ComfyUI/models/checkpoints/SDXL/juggernautXL_ragnarokBy.safetensors`
- Réseau : rank 72, alpha 36, AdamW8bit, `no_half_vae = true`, mixed precision fp16
- Steps : 6000, sauvegarde tous les 300 steps

## Logs

- TensorBoard : `training_output/nemeris_sdxl/tb/20261027065212/nemeriys_sdxl/events.out.tfevents.1793080357.advanux.1667316.0`
- Log texte : `training_output/nemeris_sdxl/nemeriys_sdxl-v2.log` (0 octet ; les scalaires sont dans TensorBoard)
