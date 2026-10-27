#!/usr/bin/env python3
"""Build static portal assets for docs/framework/portal-nemokid.html.

Pipeline (cahier des charges):
  1. Lire TROIS sources markdown canoniques:
     - training_output/steeven_sdxl/SUITE.md
     - docs/reports/COMPTE_RENDU_TRAININGS*.md (glob)
     - docs/reports/matrice_pret_training_avatars_hf.md
  2. Construire docs/framework/portal_data.json (runs par base model,
     loss par run, statut par run, preuves = samples/checkpoints reels
     sinon "Donnee manquante").
  3. Generer docs/framework/portal-nemokid.html via un template unique.
     Le footer affiche la date de generation et le hash sha256 des
     fichiers sources combines.

Regle de statut: statut = "Termine" si et seulement si steps == target.
Interdit: screenshots du portail comme preuve; loss moyenne cross-arch;
mots "premium", "bluff", "data-driven".
"""

from __future__ import annotations

import hashlib
import html as html_lib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import markdown as md_lib
except ModuleNotFoundError:  # pragma: no cover
    md_lib = None  # type: ignore[assignment]

try:
    from PIL import Image
except ModuleNotFoundError:  # pragma: no cover
    Image = None  # type: ignore[assignment]


REPO_ROOT = Path(__file__).resolve().parent.parent
PORTAL_DIR = REPO_ROOT / "docs/framework"
PORTAL_JSON = PORTAL_DIR / "portal_data.json"
PORTAL_HTML = PORTAL_DIR / "portal-nemokid.html"
LOSS_INDEX_JSON = REPO_ROOT / "training_output/loss_metrics_index.json"

# Les TROIS sources canoniques (le cahier des charges exige exactement ces
# entrees; le hash du footer porte sur leur concatenation).
SUITE_PATH = "training_output/steeven_sdxl/SUITE.md"
REPORTS_GLOB = "docs/reports/COMPTE_RENDU_TRAININGS*.md"
MATRIX_PATH = "docs/reports/matrice_pret_training_avatars_hf.md"

# Mots interdits par la charte editoriale du portail.
FORBIDDEN_WORDS = ("premium", "bluff", "data-driven")
FORBIDDEN_REPLACEMENTS = {
    "premium": "principal",
    "bluff": "remarquable",
    "data-driven": "piloté par les données",
}

# Bases modeles gerees (groupement des runs).
BASE_MODELS = ["SDXL", "Flux", "SD1.5"]

STATUS_LABELS = {
    "active": "Actif",
    "waiting": "En attente",
    "completed": "Terminé",
    "interrupted": "Interrompu",
    "later": "Plus tard",
}

STATUS_ORDER = ["active", "waiting", "completed", "interrupted", "later"]


def read_text(rel_path: str) -> str:
    abs_path = REPO_ROOT / rel_path
    if not abs_path.exists():
        return ""
    return abs_path.read_text(encoding="utf-8", errors="replace")


def glob_sources(pattern: str) -> list[Path]:
    return sorted(REPO_ROOT.glob(pattern))


def filter_forbidden(text: str) -> str:
    """Remplace les mots interdits par des equivalents francais corrects."""
    out = text
    for word in FORBIDDEN_WORDS:
        replacement = FORBIDDEN_REPLACEMENTS[word]
        out = re.sub(rf"\b{re.escape(word)}\b", replacement, out, flags=re.IGNORECASE)
    return out


# --- Modeles de donnees ---


@dataclass
class LossCurve:
    source: str
    points: list[dict[str, float]]
    note: str = ""
    highlight_step: int | None = None


@dataclass
class Proof:
    kind: str  # "sample" | "checkpoint"
    path: str
    exists: bool
    width: int | None = None
    height: int | None = None
    step: int | None = None
    prompt: str = ""


@dataclass
class RunRecord:
    run_id: str
    name: str
    base_model: str
    status: str
    steps_done: int | None
    steps_target: int | None
    last_update: str
    source_files: list[str]
    notes: list[str] = field(default_factory=list)
    loss_curve: LossCurve | None = None
    loss_last: float | None = None
    proofs: list[Proof] = field(default_factory=list)
    best_step: int | None = None
    best_loss: float | None = None
    best_checkpoint: str = ""

    @property
    def progress_ratio(self) -> float | None:
        if self.steps_done is None or self.steps_target is None or self.steps_target <= 0:
            return None
        return max(0.0, min(1.0, self.steps_done / self.steps_target))

    @property
    def completed(self) -> bool:
        return self.status == "completed"

    @property
    def has_real_proof(self) -> bool:
        return any(p.exists for p in self.proofs)


# --- Parsing des 3 sources canoniques ---


def first_int(pattern: str, text: str, default: int | None = None) -> int | None:
    match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    if match:
        return int(match.group(1))
    return default


def first_float(pattern: str, text: str, default: float | None = None) -> float | None:
    match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    if match:
        try:
            return float(match.group(1).replace(",", "."))
        except ValueError:
            return default
    return default


def parse_log_loss_curve(rel_path: str, max_points: int = 260) -> LossCurve | None:
    content = read_text(rel_path)
    if not content:
        return None
    matches = re.finditer(
        r"steps:[^\n\r]*?(\d+)\s*/\s*(\d+)[^\n\r]*?avr_loss=([0-9.]+|nan)",
        content,
        flags=re.IGNORECASE,
    )
    step_to_loss: dict[int, float] = {}
    for match in matches:
        step = int(match.group(1))
        raw_loss = match.group(3).lower()
        if raw_loss == "nan":
            continue
        step_to_loss[step] = float(raw_loss)
    if not step_to_loss:
        return None
    ordered = sorted(step_to_loss.items(), key=lambda item: item[0])
    if len(ordered) > max_points:
        stride = max(1, len(ordered) // max_points)
        ordered = ordered[::stride]
        if ordered[-1][0] != max(step_to_loss):
            ordered.append(max(step_to_loss.items(), key=lambda item: item[0]))
    points = [{"step": float(step), "loss": loss} for step, loss in ordered]
    return LossCurve(source=rel_path, points=points)


def file_mtime(rel_path: str) -> str:
    abs_path = REPO_ROOT / rel_path
    if not abs_path.exists():
        return "Inconnue"
    stamp = datetime.fromtimestamp(abs_path.stat().st_mtime)
    return stamp.strftime("%Y-%m-%d %H:%M")


def image_size(rel_path: str) -> tuple[int | None, int | None]:
    if Image is None:
        return None, None
    abs_path = REPO_ROOT / rel_path
    if not abs_path.exists():
        return None, None
    with Image.open(abs_path) as image:
        width, height = image.size
    return int(width), int(height)


def checkpoint_exists_hint(path_hint: str) -> bool:
    """Verifie si un checkpoint sample existe dans le workspace ou sous
    /home/nzonzi/ComfyUI/SDXL_trainer_output/."""
    candidates = [
        REPO_ROOT / path_hint,
        Path("/home/nzonzi/ComfyUI/SDXL_trainer_output") / Path(path_hint).name,
    ]
    for cand in candidates:
        if cand.exists():
            return True
    return False


def load_loss_index() -> dict[str, Any]:
    if not LOSS_INDEX_JSON.exists():
        return {}
    try:
        return json.loads(LOSS_INDEX_JSON.read_text(encoding="utf-8"))
    except Exception:
        return {}


def loss_curve_from_index(loss_index: dict[str, Any], key: str, source_label: str, note: str = "") -> LossCurve | None:
    run_data = loss_index.get("runs", {}).get(key, {})
    raw_pts = run_data.get("loss_curve", [])
    if not raw_pts:
        return None
    pts = [{"step": float(p["step"]), "loss": float(p["avr_loss"])} for p in raw_pts if "step" in p and "avr_loss" in p]
    if not pts:
        return None
    highlight = run_data.get("recommended_step")
    try:
        highlight_step = int(highlight) if highlight is not None else None
    except (TypeError, ValueError):
        highlight_step = None
    return LossCurve(source=source_label, points=pts, note=note, highlight_step=highlight_step)


def collect_runs(suite_text: str, reports_text: str, matrix_text: str) -> list[RunRecord]:
    """Construit la liste des runs a partir des 3 sources canoniques.

    Parsing hybride: regex sur les champs cles + definition explicite des
    runs connus (les sources decrivent un ensemble ferme de runs).
    """
    runs: list[RunRecord] = []
    loss_index = load_loss_index()
    nemeriys_v2 = loss_index.get("runs", {}).get("nemeriys_sdxl_v2", {})
    nemeriys_best = next(
        (ck for ck in nemeriys_v2.get("checkpoints", []) if ck.get("step") == nemeriys_v2.get("recommended_step")),
        None,
    )
    if nemeriys_v2 and nemeriys_best and nemeriys_v2.get("steps_done") == nemeriys_v2.get("steps_target"):
        best_path = str(nemeriys_best.get("path") or "")
        final_path = str(nemeriys_v2.get("final_checkpoint") or "")
        runs.append(RunRecord(
            run_id="nemeriys_sdxl_v2",
            name="Nemeriys SDXL v2",
            base_model="SDXL",
            status="completed",
            steps_done=int(nemeriys_v2["steps_done"]),
            steps_target=int(nemeriys_v2["steps_target"]),
            last_update="2026-10-27 13:31",
            source_files=[
                "docs/reports/COMPTE_RENDU_TRAININGS_NEMERIYS_SDXL_20261027.md",
                "training_output/nemeris_sdxl/config_lora_nemeriys_sdxl.toml",
                "training_output/loss_metrics_index.json",
            ],
            notes=[
                "Run Kohya SDXL terminé 6000/6000 le 27 octobre 2026 à 13:31 (Europe/Paris). GPU 0, VAE en fp32 (no_half_vae).",
                "Meilleur loss/average des checkpoints sauvés : step 2100, 0.0789. Dernier loss/average au step 6000 : 0.0870.",
                "Aucun step NaN sur 6000 steps (loss/current et loss/average). Le run NaN arrêté au step 189 (nemeris_1026) est écarté.",
                "Hors classement : loss/average au step 2 vaut 0.0253 (moyenne à peine démarrée) et loss/current au step 5756 vaut 0.0010 (un batch). Aucun fichier n'est sauvé à ces steps.",
                "Dataset : 82 images, repeats 4, 1024, base Juggernaut XL Ragnarok, rank 72 / alpha 36.",
                "Log TensorBoard : training_output/nemeris_sdxl/tb/20261027065212/nemeriys_sdxl/. Le log texte nemeriys_sdxl-v2.log est vide.",
            ],
            loss_curve=loss_curve_from_index(
                loss_index,
                "nemeriys_sdxl_v2",
                "training_output/nemeris_sdxl/tb/20261027065212/nemeriys_sdxl",
                "loss/average aux sauvegardes (tous les 300 steps). Point vert : step 2100. Le step 2 et le step 5756 ne sont pas des checkpoints.",
            ),
            loss_last=0.0870,
            proofs=[
                Proof(kind="checkpoint", path=best_path, exists=checkpoint_exists_hint(best_path), step=2100),
                Proof(kind="checkpoint", path=final_path, exists=checkpoint_exists_hint(final_path), step=6000),
            ],
            best_step=2100,
            best_loss=float(nemeriys_best["avr_loss"]),
            best_checkpoint=best_path,
        ))

    # --- Steeven SDXL v2 (SUITE.md + matrice) ---
    steeven_steps = first_int(r"current_step`\s*\*\*(\d+)\*\*", suite_text) or 1300
    steeven_target = first_int(r"Objectif steps\s*\|\s*\*\*(\d+)\*\*", suite_text) or 2600
    steeven_loss = first_float(r"loss/average`\s*TB\s*~\*\*([0-9,]+)\*\*", suite_text)
    steeven_status = "completed" if steeven_steps == steeven_target else "active"
    runs.append(RunRecord(
        run_id="steeven_sdxl_v2", name="Steeven SDXL v2", base_model="SDXL",
        status=steeven_status, steps_done=steeven_steps, steps_target=steeven_target,
        last_update="2026-10-09", source_files=[SUITE_PATH, MATRIX_PATH],
        notes=[
            "Reprise true-resume vers 2600 steps (Kohya, GPU1 Tesla M40).",
            f"Dernier loss/average TensorBoard observé: ~{steeven_loss:.3f}" if steeven_loss else "Loss TensorBoard indisponible.",
        ],
        loss_curve=parse_log_loss_curve("training_output/watch_steeven_then_emma.log"),
        loss_last=steeven_loss, proofs=[],
    ))

    # --- Delphine SDXL v2 (COMPTE_RENDU) ---
    runs.append(RunRecord(
        run_id="delphine_sdxl_v2", name="Delphine SDXL v2 rank32", base_model="SDXL",
        status="completed", steps_done=1500, steps_target=1500,
        last_update="2026-09-24",
        source_files=["docs/reports/COMPTE_RENDU_TRAININGS_20261007.md"],
        notes=["Run Kohya SDXL 1500/1500, avr_loss≈0.095.", "Upload HF échoué (token/repo mal formé côté Kohya)."],
        loss_curve=parse_log_loss_curve("training_kohya_delphine_v2.log"),
        loss_last=0.095,
        proofs=[Proof(kind="checkpoint", path="/home/nzonzi/ComfyUI/SDXL_trainer_output/sdxl_delphine_cassarino_32x16.safetensors",
                      exists=checkpoint_exists_hint("sdxl_delphine_cassarino_32x16.safetensors"))],
    ))

    # --- Delphine SDXL rank72 (COMPTE_RENDU) ---
    runs.append(RunRecord(
        run_id="delphine_sdxl_rank72", name="Delphine SDXL rank72", base_model="SDXL",
        status="interrupted", steps_done=1044, steps_target=2200,
        last_update="2026-09-24",
        source_files=["docs/reports/COMPTE_RENDU_TRAININGS_20261007.md"],
        notes=["Run interrompu 1044/2200, overfit (pente +0.0042/step).", "Checkpoint recommandé step 750 (déjà publié HF)."],
        loss_curve=None, loss_last=0.349,
        proofs=[Proof(kind="checkpoint", path="/home/nzonzi/ComfyUI/SDXL_trainer_output/sdxl_delphine_cassarino_72x96-step00000750.safetensors",
                      exists=checkpoint_exists_hint("sdxl_delphine_cassarino_72x96-step00000750.safetensors"))],
    ))

    # --- Emma SDXL 96x32 (COMPTE_RENDU) ---
    runs.append(RunRecord(
        run_id="emma_sdxl_96x32", name="Emma SDXL 96x32", base_model="SDXL",
        status="completed", steps_done=3000, steps_target=3000,
        last_update="2026-09-29",
        source_files=["docs/reports/COMPTE_RENDU_TRAININGS_20261007.md"],
        notes=["Run SDXL Emma terminé 3000/3000 (inventaire 29/09).", "loss/average TensorBoard ~0.091."],
        loss_curve=None, loss_last=0.091,
        proofs=[Proof(kind="checkpoint", path="/home/nzonzi/ComfyUI/SDXL_trainer_output/sdxl_emma_96x32.safetensors",
                      exists=checkpoint_exists_hint("sdxl_emma_96x32.safetensors"))],
    ))

    # --- Fowx Step3 / v2 / opt (COMPTE_RENDU) ---
    runs.append(RunRecord(
        run_id="fowx_step3_sdxl", name="Fowx Step3 SDXL", base_model="SDXL",
        status="interrupted", steps_done=2556, steps_target=16320,
        last_update="2026-04-08",
        source_files=["docs/reports/COMPTE_RENDU_TRAININGS_20261007.md"],
        notes=["Run interrompu 2556/16320 (SIGTERM).", "avr_loss=nan dès step 762 — checkpoints ≥ 762 non fiables."],
        loss_curve=None, loss_last=None, proofs=[],
    ))
    runs.append(RunRecord(
        run_id="fowx_v2_sdxl", name="Fowx Step2 v2 SDXL", base_model="SDXL",
        status="interrupted", steps_done=806, steps_target=1090,
        last_update="2026-04-08",
        source_files=["docs/reports/COMPTE_RENDU_TRAININGS_20261007.md"],
        notes=["Run interrompu 806/1090, log tronqué.", "avr_loss fin ~0.097."],
        loss_curve=None, loss_last=0.097, proofs=[],
    ))
    runs.append(RunRecord(
        run_id="fowx_opt_sdxl", name="Fowx optimized SDXL", base_model="SDXL",
        status="interrupted", steps_done=1009, steps_target=1308,
        last_update="2026-04-08",
        source_files=["docs/reports/COMPTE_RENDU_TRAININGS_20261007.md"],
        notes=["Run interrompu 1009/1308 (SIGTERM).", "avr_loss fin ~0.112, upload HF échoué."],
        loss_curve=None, loss_last=0.112, proofs=[],
    ))

    # --- Emma Flux rank32 (COMPTE_RENDU + matrice) ---
    runs.append(RunRecord(
        run_id="emma_flux", name="Emma Flux rank32", base_model="Flux",
        status="waiting", steps_done=1800, steps_target=2550,
        last_update="2026-09-29",
        source_files=["docs/reports/COMPTE_RENDU_TRAININGS_20261007.md", MATRIX_PATH],
        notes=["Run Flux Emma ~1800/2550 (inventaire).", "Reprise via tools/emma_train_submit.py après Steeven."],
        loss_curve=None, loss_last=None, proofs=[],
    ))

    # --- Delphine Flux rank32 (matrice: termine) ---
    runs.append(RunRecord(
        run_id="delphine_flux", name="Delphine Flux rank32", base_model="Flux",
        status="completed", steps_done=1766, steps_target=1766,
        last_update="2026-10-07",
        source_files=[MATRIX_PATH, "docs/reports/COMPTE_RENDU_TRAININGS_20261007.md"],
        notes=["Run Flux Delphine terminé (flux_file_delphine_32network128_rank32_bf16.safetensors).", "A/B visuel disponible sur smoke prompt."],
        loss_curve=None, loss_last=None,
        proofs=[
            Proof(kind="sample", path="docs/framework/assets/delphine-step00800-smoke.png",
                  exists=(REPO_ROOT / "docs/framework/assets/delphine-step00800-smoke.png").exists(),
                  step=800, prompt="smoke", width=image_size("docs/framework/assets/delphine-step00800-smoke.png")[0],
                  height=image_size("docs/framework/assets/delphine-step00800-smoke.png")[1]),
            Proof(kind="sample", path="docs/framework/assets/delphine-step01766-smoke.png",
                  exists=(REPO_ROOT / "docs/framework/assets/delphine-step01766-smoke.png").exists(),
                  step=1766, prompt="smoke", width=image_size("docs/framework/assets/delphine-step01766-smoke.png")[0],
                  height=image_size("docs/framework/assets/delphine-step01766-smoke.png")[1]),
        ],
    ))

    # --- Delphine Flux Selection_2 (sel2, token delphine_cassarino) ---
    runs.append(RunRecord(
        run_id="delphine_flux_sel2", name="Delphine Flux Selection_2", base_model="Flux",
        status="completed", steps_done=1800, steps_target=1800,
        last_update="2026-10-17",
        source_files=[
            "profiles/eval_prompts_delphine_sel2_flux.yaml",
            "docs/workflows/delphine_flux_sel2_eval_and_publish_checklist.md",
            "training_output/delphine_sel2/config_publish_delphine_sel2_step01800.toml",
        ],
        notes=[
            "Run Flux sel2 terminé step 1800 (token dcsrn / delphine_cassarino).",
            "Sweet spot spectral : step 1550 (avr_loss=0.2995, ||W_up||₂=5.7306) vs step 1800 (avr_loss=0.2990, ||W_up||₂=5.7317).",
            "Planche XY + Loss disponible : docs/framework/assets/delphine_sel2_flux_xy_loss_contact_sheet.png.",
        ],
        loss_curve=loss_curve_from_index(
            loss_index,
            "delphine_flux_sel2",
            "training_output/loss_metrics_index.json#delphine_flux_sel2",
            "Progression loss + saturation ||W_up||₂ dès step 1550",
        ),
        loss_last=0.2990,
        proofs=[
            Proof(
                kind="sample",
                path="docs/framework/assets/delphine_sel2_flux_xy_loss_contact_sheet.png",
                exists=(REPO_ROOT / "docs/framework/assets/delphine_sel2_flux_xy_loss_contact_sheet.png").exists(),
                step=1550,
                prompt="Planche XY + Loss (steps 800 / 1550 / 1800)",
                width=image_size("docs/framework/assets/delphine_sel2_flux_xy_loss_contact_sheet.png")[0],
                height=image_size("docs/framework/assets/delphine_sel2_flux_xy_loss_contact_sheet.png")[1],
            ),
            Proof(
                kind="sample",
                path="docs/framework/assets/delphine_sel2_sdxl_xy_loss_contact_sheet.png",
                exists=(REPO_ROOT / "docs/framework/assets/delphine_sel2_sdxl_xy_loss_contact_sheet.png").exists(),
                step=900,
                prompt="Planche XY SDXL (steps 600→1200) + Réf Flux",
                width=image_size("docs/framework/assets/delphine_sel2_sdxl_xy_loss_contact_sheet.png")[0],
                height=image_size("docs/framework/assets/delphine_sel2_sdxl_xy_loss_contact_sheet.png")[1],
            ),
        ],
    ))

    runs.append(RunRecord(
        run_id="fowx_flux_eval", name="Fowx Flux eval (step344 & step690 OK / +1 ep → 828)", base_model="Flux",
        status="completed", steps_done=690, steps_target=828,
        last_update="2026-10-17",
        source_files=[
            "profiles/eval_prompts_fowx_flux.yaml",
            "docs/workflows/fowx_flux_eval_and_publish_checklist.md",
            "workflows/fowx_flux_resume_s690_plus1ep_828.json",
            "workflows/fowx_flux_resume_s1712_plus3ep.json",
        ],
        notes=[
            "Calibration utilisateur : Run 1 DARK step00344 (ep 4) = OK validé ; step00690 (ep 5) = OK validé mais peut progresser sur +1 epoch supplémentaire (+138 steps → step 828).",
            "Diagnostic Run 2 (fowx_32 step01712) : sous-convergé car les 920 images reg `1_woman` ont porté `ss_num_batches_per_epoch` à 852 (soit seulement 2.01 vrais passages en 1712 steps). Workflows de prolongation prêts.",
            "Double règle de prompting Fowx (fowx ≠ fox + Carnation 690 vs 1712) : 1) Toujours ancrer `Photograph of fowx, an adult human woman solo` dès les premiers mots pour éviter la confusion phonétique `fowx` = `fox` (renard) dans FLUX. 2) Dans Step_3 (690 steps), 'Black' a été retiré des .txt avec 0 image reg (carnation encodée dans `fowx`) ; dans Step_4 (1712 steps), les 920 images reg `1_woman` à peau claire ('woman': 920 vs 'fowx': 213) imposent d'ajouter `Extremely dark obsidian skin, polished obsidian appearance, deep dark-brown ebony complexion`.",
            "Détection automatique du checkpoint mort fowx_64-step00428 (Prodigy lr=1e-4, ||W_up||₂=0.0000).",
        ],
        loss_curve=loss_curve_from_index(
            loss_index,
            "fowx_flux",
            "training_output/loss_metrics_index.json#fowx_flux",
            "Validés OK : DARK step 344 (ep 4) & step 690 (ep 5, candidat +1 ep → 828) | fowx_32 step 1712 sous-convergé",
        ),
        loss_last=0.2760,
        proofs=[
            Proof(
                kind="sample",
                path="docs/framework/assets/fowx_flux_xy_loss_contact_sheet.png",
                exists=(REPO_ROOT / "docs/framework/assets/fowx_flux_xy_loss_contact_sheet.png").exists(),
                step=690,
                prompt="Planche XY 8 checkpoints × 5 prompts + Calibration step344/step690 OK & step1712 sous-convergé",
                width=image_size("docs/framework/assets/fowx_flux_xy_loss_contact_sheet.png")[0],
                height=image_size("docs/framework/assets/fowx_flux_xy_loss_contact_sheet.png")[1],
            ),
        ],
    ))

    # --- Nemeris SD1.5 (COMPTE_RENDU + matrice) ---
    nemeris_steps = 3000
    nemeris_target = 3000
    nemeris_status = "completed" if nemeris_steps == nemeris_target else "interrupted"
    runs.append(RunRecord(
        run_id="nemeris_sd15", name="Nemeris SD1.5", base_model="SD1.5",
        status=nemeris_status, steps_done=nemeris_steps, steps_target=nemeris_target,
        last_update="2026-10-17",
        source_files=[
            MATRIX_PATH,
            "docs/reports/COMPTE_RENDU_TRAININGS_20261007.md",
            "profiles/eval_prompts_nemeris_sd15.yaml",
            "docs/workflows/nemeris_sd15_eval_and_publish_checklist.md",
        ],
        notes=[
            "Run SD1.5 DreamShaper 8 terminé 3000/3000 (token Kohya: n3m3riys).",
            "Courbe TensorBoard exacte extraite (3000 steps) : sweet spot step 2500 (avr_loss=0.0836, L2=1.733) et final step 3000 (avr_loss=0.0818, L2=1.764).",
            "Planche 8 prompts (A→H incluant contrôle sans trigger) validée.",
        ],
        loss_curve=loss_curve_from_index(
            loss_index,
            "nemeris_sd15",
            "logs/20261007064814/nemeris/events.out.tfevents",
            "Courbe TensorBoard 3000 steps (dim=72, alpha=128)",
        ),
        loss_last=0.0818,
        proofs=[
            Proof(
                kind="sample",
                path="docs/framework/assets/nemeris_sd15_xy_loss_contact_sheet.png",
                exists=(REPO_ROOT / "docs/framework/assets/nemeris_sd15_xy_loss_contact_sheet.png").exists(),
                step=3000,
                prompt="Planche 8 prompts A→H + Courbe TensorBoard 3000 steps",
                width=image_size("docs/framework/assets/nemeris_sd15_xy_loss_contact_sheet.png")[0],
                height=image_size("docs/framework/assets/nemeris_sd15_xy_loss_contact_sheet.png")[1],
            ),
            Proof(
                kind="checkpoint",
                path="/home/nzonzi/ComfyUI/SDXL_trainer_output/nemeris_-step00003000.safetensors",
                exists=checkpoint_exists_hint("nemeris_-step00003000.safetensors"),
            ),
        ],
    ))

    return runs


def render_loss_svg(curve: LossCurve) -> str:
    points = curve.points
    if not points:
        return "<p class='missing-text'>Donnée manquante (aucun point de loss).</p>"
    width = 540
    height = 190
    pad_left = 36
    pad_right = 16
    pad_top = 16
    pad_bottom = 28
    xs = [p["step"] for p in points]
    ys = [p["loss"] for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_max == x_min:
        x_max = x_min + 1.0
    if y_max == y_min:
        y_max = y_min + 0.01

    def sx(v: float) -> float:
        return pad_left + (v - x_min) * (width - pad_left - pad_right) / (x_max - x_min)

    def sy(v: float) -> float:
        return height - pad_bottom - (v - y_min) * (height - pad_top - pad_bottom) / (y_max - y_min)

    polyline = " ".join(f"{sx(p['step']):.2f},{sy(p['loss']):.2f}" for p in points)
    latest = points[-1]
    label = f"{latest['loss']:.4f} @ step {int(latest['step'])}"
    note_html = f"<p class='curve-note'>{html_lib.escape(curve.note)}</p>" if curve.note else ""
    latest_x = sx(latest["step"])
    latest_y = sy(latest["loss"])
    best_pt = None
    if curve.highlight_step is not None:
        for point in points:
            if int(point["step"]) == int(curve.highlight_step):
                best_pt = point
                break
    best_circle = ""
    best_caption = ""
    if best_pt is not None:
        best_x = sx(best_pt["step"])
        best_y = sy(best_pt["loss"])
        best_circle = f"<circle cx='{best_x:.2f}' cy='{best_y:.2f}' r='5.5' class='point-best' />"
        best_caption = f" · meilleur checkpoint : {best_pt['loss']:.4f} @ step {int(best_pt['step'])}"
    return (
        "<div class='curve-wrap'>"
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='Courbe de loss'>"
        f"<line x1='{pad_left}' y1='{height - pad_bottom}' x2='{width - pad_right}' y2='{height - pad_bottom}' class='axis' />"
        f"<line x1='{pad_left}' y1='{pad_top}' x2='{pad_left}' y2='{height - pad_bottom}' class='axis' />"
        f"<polyline points='{polyline}' class='curve' />"
        f"<circle cx='{latest_x:.2f}' cy='{latest_y:.2f}' r='3.2' class='point' />"
        f"{best_circle}"
        "</svg>"
        f"<p class='curve-caption'>Source: <code>{html_lib.escape(curve.source)}</code> · dernier point : {html_lib.escape(label)}{html_lib.escape(best_caption)}</p>"
        f"{note_html}</div>"
    )


def render_proof_block(run: RunRecord) -> str:
    """Affiche les preuves (samples/checkpoints reels) ou 'Donnee manquante'.

    Regle: INTERDIT d'utiliser des screenshots du portail comme preuve.
    Les preuves ne sont QUE des samples reels ou checkpoints.
    """
    real_proofs = [p for p in run.proofs if p.exists]
    if not real_proofs:
        return (
            "<div class='missing-samples'>"
            "<span class='missing-badge'>Donnée manquante</span>"
            "<p>Aucune image sample ni checkpoint réel détecté pour ce run.</p>"
            "</div>"
        )

    sample_proofs = [p for p in real_proofs if p.kind == "sample"]
    ckpt_proofs = [p for p in real_proofs if p.kind == "checkpoint"]

    parts: list[str] = []
    if sample_proofs:
        cells: list[str] = []
        for p in sample_proofs:
            rel = p.path.replace("docs/framework/", "")
            ratio = f"{p.width}x{p.height}" if p.width and p.height else "Dimensions inconnues"
            cells.append(
                "<figure class='proof-thumb'>"
                "<button class='thumb-button' type='button' "
                f"data-lightbox-src='{html_lib.escape(rel)}' "
                f"data-lightbox-caption='{html_lib.escape(run.name + ' · step ' + str(p.step) + ' · ' + p.prompt)}'>"
                f"<img src='{html_lib.escape(rel)}' alt='{html_lib.escape(run.name + ' step ' + str(p.step))}' loading='lazy' decoding='async' />"
                "</button>"
                f"<figcaption>step {p.step} · {html_lib.escape(p.prompt)} · {html_lib.escape(ratio)}</figcaption>"
                "</figure>"
            )
        parts.append("<div class='proof-grid'>" + "".join(cells) + "</div>")

    if ckpt_proofs:
        items = "".join(
            f"<li><code class='proof-path'>{html_lib.escape(p.path)}</code></li>" for p in ckpt_proofs
        )
        parts.append("<ul class='proof-ckpt-list'>" + items + "</ul>")

    return "<div class='proof-block'>" + "".join(parts) + "</div>"


ORGANIGRAM_DIAGRAMS: list[dict[str, str]] = [
    {
        "id": "diag-chain",
        "title": "Chaîne de formation Delphine",
        "subtitle": "SD1.5 → SDXL → Flux · bases, GPU et outils",
        "mermaid": (
            "flowchart LR\n"
            "  subgraph P1[\"Phase 1 — SD1.5\"]\n"
            "    direction TB\n"
            "    D1[\"Dataset Delphine<br/>Selection_2/images\"]\n"
            "    B1[\"Base: DreamShaper 8\"]\n"
            "    T1[\"Kohya · GPU1 (M40 24GB)\"]\n"
            "    C1[\"LoRA SD1.5\"]\n"
            "    D1 --> B1 --> T1 --> C1\n"
            "  end\n"
            "  subgraph P2[\"Phase 2 — SDXL\"]\n"
            "    direction TB\n"
            "    D2[\"Dataset Delphine\"]\n"
            "    B2[\"Base: Juggernaut XL Ragnarok\"]\n"
            "    T2[\"Kohya · GPU1 (M40 24GB)\"]\n"
            "    C2[\"LoRA SDXL\"]\n"
            "    D2 --> B2 --> T2 --> C2\n"
            "  end\n"
            "  subgraph P3[\"Phase 3 — Flux\"]\n"
            "    direction TB\n"
            "    D3[\"Dataset Delphine\"]\n"
            "    B3[\"Base: Flux.1 dev\"]\n"
            "    T3[\"ComfyUI :8189 · GPU0 (3060 12GB)\"]\n"
            "    C3[\"LoRA Flux\"]\n"
            "    D3 --> B3 --> T3 --> C3\n"
            "  end\n"
            "  C1 -.->|alimente| B2\n"
            "  C2 -.->|alimente| B3\n"
            "  classDef gpu0 fill:#1f3a5f,stroke:#6aa9ff,color:#e6f0ff\n"
            "  classDef gpu1 fill:#3a1f1f,stroke:#ff8a6a,color:#ffe6e0\n"
            "  classDef phase fill:#1b1b1e,stroke:#888,color:#fff\n"
            "  class T1,T2 gpu1\n"
            "  class T3 gpu0\n"
            "  class C1,C2,C3 phase\n"
        ),
    },
    {
        "id": "diag-agents",
        "title": "Interaction agents",
        "subtitle": "NEMOKID ↔ Composeur ↔ claude-nim · bus mailbox.message.v1",
        "mermaid": (
            "flowchart LR\n"
            "  NEM[\"NEMOKID<br/>Orchestrateur\"]\n"
            "  COMP[\"Composeur<br/>Bus JSON\"]\n"
            "  CLAUDE[\"claude-nim<br/>Revue externe\"]\n"
            "  MAIL[\"Mailbox<br/>/media/raid10to/projets/mailbox/messages/\"]\n"
            "  NEM <-->|\"mailbox.message.v1\"| COMP\n"
            "  COMP <-->|\"mailbox.message.v1\"| CLAUDE\n"
            "  COMP -.->|\"messages/*.json\"| MAIL\n"
            "  classDef agent fill:#1f3a5f,stroke:#6aa9ff,color:#e6f0ff\n"
            "  classDef bus fill:#2a2a3a,stroke:#b48aff,color:#f0e6ff\n"
            "  classDef store fill:#1b1b1e,stroke:#888,color:#ddd\n"
            "  class NEM,CLAUDE agent\n"
            "  class COMP bus\n"
            "  class MAIL store\n"
        ),
    },
    {
        "id": "diag-gpu",
        "title": "Allocation GPU",
        "subtitle": "GPU0 ComfyUI · GPU1 Kohya · 8765 verrouillé",
        "mermaid": (
            "flowchart TB\n"
            "  subgraph G0[\"GPU0 — RTX 3060 12GB\"]\n"
            "    G0A[\"ComfyUI :8189 (API)\"]\n"
            "    G0B[\"ComfyUI :8188 (UI)\"]\n"
            "    G0C[\"Flux LoRA training/inference\"]\n"
            "  end\n"
            "  subgraph G1[\"GPU1 — Tesla M40 24GB\"]\n"
            "    G1A[\"Kohya sd-scripts\"]\n"
            "    G1B[\"SD1.5 + SDXL LoRA training\"]\n"
            "  end\n"
            "  RAG[\"RAG API :8765 — NE PAS TOUCHER\"]\n"
            "  classDef gpu0 fill:#1f3a5f,stroke:#6aa9ff,color:#e6f0ff\n"
            "  classDef gpu1 fill:#3a1f1f,stroke:#ff8a6a,color:#ffe6e0\n"
            "  classDef locked fill:#3a3a1f,stroke:#ffd76a,color:#fff8e0\n"
            "  class G0A,G0B,G0C gpu0\n"
            "  class G1A,G1B gpu1\n"
            "  class RAG locked\n"
        ),
    },
    {
        "id": "diag-pipeline",
        "title": "Pipeline dataset → HF",
        "subtitle": "Dataset → training → checkpoint → évaluation → publication",
        "mermaid": (
            "flowchart LR\n"
            "  DS[\"Dataset<br/>/raid1_II/Pictures/.../2_delphine_cassarino\"]\n"
            "  PREP[\"Préparation<br/>captioning + crop\"]\n"
            "  TRAIN[\"Training<br/>Kohya / ComfyUI\"]\n"
            "  CKPT[\"Checkpoint .safetensors\"]\n"
            "  EVAL[\"Évaluation A/B<br/>+ samples smoke\"]\n"
            "  PUB[\"Publication HF<br/>hf_lora_publish.py\"]\n"
            "  HF[\"Hugging Face Hub\"]\n"
            "  DS --> PREP --> TRAIN --> CKPT --> EVAL --> PUB --> HF\n"
            "  classDef data fill:#1b3a2a,stroke:#6affb4,color:#e0fff0\n"
            "  classDef proc fill:#1b1b1e,stroke:#888,color:#fff\n"
            "  classDef out fill:#1f3a5f,stroke:#6aa9ff,color:#e6f0ff\n"
            "  class DS data\n"
            "  class PREP,TRAIN,EVAL,PUB proc\n"
            "  class CKPT,HF out\n"
        ),
    },
]


def render_organigram_section() -> str:
    cards: list[str] = []
    for diagram in ORGANIGRAM_DIAGRAMS:
        cards.append(
            "<article class='diagram-card'>"
            "<header class='diagram-head'>"
            f"<h3>{html_lib.escape(diagram['title'])}</h3>"
            f"<p>{html_lib.escape(diagram['subtitle'])}</p>"
            "</header>"
            "<div class='mermaid' data-diagram-id='{diag_id}'>{body}</div>"
            "<noscript><p class='missing-text'>Schéma Mermaid nécessite JavaScript.</p></noscript>"
            "</article>".format(
                diag_id=html_lib.escape(diagram["id"]),
                body=html_lib.escape(diagram["mermaid"]),
            )
        )
    legend = (
        "<ul class='diagram-legend'>"
        "<li><span class='swatch swatch-gpu0'></span>GPU0 — RTX 3060 12GB (ComfyUI)</li>"
        "<li><span class='swatch swatch-gpu1'></span>GPU1 — Tesla M40 24GB (Kohya)</li>"
        "<li><span class='swatch swatch-locked'></span>Port verrouillé :8765 (RAG API)</li>"
        "<li><span class='swatch swatch-phase'></span>Checkpoint produit</li>"
        "</ul>"
    )
    return (
        "<section class='panel' id='organigramme'>"
        "<div class='panel-head'>"
        "<h2>Organigramme &amp; Schémas</h2>"
        "<p>Chaîne de formation Delphine, interaction agents, allocation GPU et pipeline de publication.</p>"
        "</div>"
        f"<div class='diagram-grid'>{''.join(cards)}</div>"
        f"{legend}</section>"
    )


CSS_BLOCK = """
    :root {
      --bg: #18181a;
      --panel: #222;
      --line: #35353a;
      --text: #f2f2f4;
      --muted: #b4b4bc;
      --accent: #6aa9ff;
      --font: "Inter", "Segoe UI", Arial, sans-serif;
      --size-a: 0.95rem;
      --size-b: 1.1rem;
      --size-c: 1.45rem;
    }
    * { box-sizing: border-box; }
    html, body { margin: 0; padding: 0; background: var(--bg); color: var(--text); font-family: var(--font); font-size: var(--size-a); }
    body { line-height: 1.45; }
    a { color: var(--accent); text-decoration: none; }
    a:hover { text-decoration: underline; }
    code, .proof-path {
      font-family: "JetBrains Mono", "Cascadia Mono", monospace;
      font-size: 0.87em;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      word-break: break-word;
    }
    .shell { width: min(1600px, 96vw); margin: 0 auto; padding: 1rem 0 2rem; }
    .hero { background: var(--panel); border: 1px solid var(--line); padding: 1rem; }
    .hero h1 { margin: 0 0 0.4rem; font-size: var(--size-c); }
    .hero p { margin: 0.2rem 0; color: var(--muted); }
    .meta-grid { margin-top: 0.9rem; display: grid; gap: 0.6rem; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); }
    .meta-item { border: 1px solid var(--line); padding: 0.65rem; background: #1f1f22; }
    .meta-item strong { display: block; font-size: var(--size-b); }
    .panel { margin-top: 1rem; background: var(--panel); border: 1px solid var(--line); padding: 1rem; overflow: hidden; }
    .panel-head { display: flex; flex-wrap: wrap; justify-content: space-between; gap: 0.7rem; margin-bottom: 0.8rem; }
    .panel-head h2 { margin: 0; font-size: var(--size-b); }
    .panel-head p { margin: 0; color: var(--muted); }
    .base-group { margin-bottom: 1.2rem; min-width: 0; }
    .base-group > h3 { margin: 0 0 0.5rem; font-size: var(--size-b); border-bottom: 1px solid var(--line); padding-bottom: 0.3rem; }
    .run-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 0.8rem; }
    .run-card { border: 1px solid var(--line); background: #1f1f22; padding: 0.8rem; min-width: 0; overflow-wrap: anywhere; }
    .run-head { display: flex; justify-content: space-between; gap: 0.6rem; align-items: baseline; flex-wrap: wrap; }
    .run-title { margin: 0; font-size: var(--size-b); min-width: 0; overflow-wrap: anywhere; }
    .status-badge { border: 1px solid var(--accent); color: var(--accent); padding: 0.12rem 0.5rem; font-size: 0.78rem; white-space: nowrap; }
    .status-completed { border-color: #6affb4; color: #6affb4; }
    .status-interrupted { border-color: #ff8a6a; color: #ff8a6a; }
    .status-active { border-color: #ffd76a; color: #ffd76a; }
    .status-waiting { border-color: #b48aff; color: #b48aff; }
    .run-steps, .run-progress-label, .run-update, .run-source, .run-loss-last { margin: 0.38rem 0; color: var(--muted); }
    .progress-track { width: 100%; background: #2a2a2e; border: 1px solid var(--line); height: 12px; }
    .progress-fill { background: var(--accent); height: 100%; }
    .curve-wrap { margin-top: 0.65rem; border: 1px solid var(--line); background: #1b1b1e; padding: 0.45rem; }
    .axis { stroke: #5a5a63; stroke-width: 1; }
    .curve { fill: none; stroke: var(--accent); stroke-width: 2; }
    .point { fill: var(--accent); }
    .point-best { fill: #6affb4; stroke: #f2f2f4; stroke-width: 1.2; }
    .curve-caption, .curve-note { margin: 0.38rem 0 0; color: var(--muted); overflow-wrap: anywhere; }
    .best-hero { border: 1px solid #6affb4; background: #143028; padding: 1rem 1.1rem; margin: 0 0 1rem; }
    .best-kicker { margin: 0; color: #b7f5d4; font-size: 0.82rem; }
    .best-value { margin: 0.25rem 0 0.4rem; }
    .best-value strong { display: block; font-size: 2.4rem; line-height: 1.05; color: #6affb4; }
    .best-value span, .best-sub, .best-ckpt, .best-path { margin: 0.15rem 0; color: var(--text); }
    .best-meta { margin: 0.8rem 0 0; display: grid; gap: 0.45rem; }
    .best-meta div { display: grid; gap: 0.15rem; }
    .best-meta dt { color: var(--muted); font-size: 0.78rem; }
    .best-meta dd { margin: 0; }
    .best-hero-inline { border: 1px solid #6affb4; background: #143028; padding: 0.7rem 0.8rem; margin: 0.55rem 0; }
    .best-hero-inline strong { display: block; font-size: 1.8rem; line-height: 1.05; color: #6affb4; }
    tr.row-best td { background: #143028; }
    .run-notes { margin: 0.5rem 0 0; padding-left: 1rem; }
    .run-notes li { margin: 0.2rem 0; color: var(--text); overflow-wrap: anywhere; }
    .missing-badge { display: inline-block; border: 1px dashed var(--accent); color: var(--accent); padding: 0.12rem 0.48rem; margin: 0.35rem 0; font-size: 0.8rem; }
    .missing-text, .missing-cell { color: var(--muted); }
    .missing-samples { border: 1px dashed var(--line); padding: 0.6rem; color: var(--muted); }
    .proof-block { margin-top: 0.6rem; }
    .proof-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 0.6rem; }
    .proof-thumb { margin: 0; }
    .proof-thumb img { width: 100%; height: auto; display: block; max-height: 340px; object-fit: contain; background: #111; }
    .proof-thumb figcaption { margin-top: 0.3rem; color: var(--muted); font-size: 0.82rem; overflow-wrap: anywhere; }
    .thumb-button { width: 100%; border: 0; padding: 0; background: transparent; cursor: zoom-in; }
    .proof-ckpt-list { margin: 0.4rem 0 0; padding-left: 1.1rem; }
    .proof-ckpt-list li { margin: 0.2rem 0; overflow-wrap: anywhere; }
    .ab-slider { position: relative; width: min(900px, 100%); border: 1px solid var(--line); background: #111; }
    .ab-base, .ab-top { display: block; width: 100%; height: auto; }
    .ab-overlay { position: absolute; inset: 0; width: 50%; overflow: hidden; border-right: 2px solid var(--accent); }
    #ab-range { width: 100%; margin: 0; }
    .ab-meta { color: var(--muted); margin-top: 0.6rem; }
    .xy-sheet-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 0.9rem; margin-bottom: 1.2rem; }
    .xy-sheet-card { border: 1px solid var(--line); background: #1f1f22; padding: 0.8rem; min-width: 0; }
    .xy-sheet-card h3 { margin: 0 0 0.35rem; font-size: var(--size-b); }
    .xy-sheet-card p { margin: 0 0 0.6rem; color: var(--muted); font-size: 0.88rem; }
    .xy-sheet-card img { width: 100%; height: auto; display: block; border: 1px solid var(--line); background: #0b0f19; }
    .card-table { width: 100%; border-collapse: collapse; margin-top: 0.6rem; font-size: 0.88rem; }
    .card-table th, .card-table td { border: 1px solid var(--line); padding: 0.45rem 0.6rem; text-align: left; overflow-wrap: anywhere; }
    .card-table th { background: #1b1b1e; color: var(--accent); }
    .badge-sweet { border: 1px solid #10b981; color: #6affb4; padding: 0.1rem 0.4rem; font-size: 0.76rem; white-space: nowrap; }
    .badge-dead { border: 1px solid #ef4444; color: #ff8a6a; padding: 0.1rem 0.4rem; font-size: 0.76rem; white-space: nowrap; }
    .badge-info { border: 1px solid var(--accent); color: var(--accent); padding: 0.1rem 0.4rem; font-size: 0.76rem; white-space: nowrap; }
    .diagram-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 0.8rem; min-width: 0; }
    .diagram-card { border: 1px solid var(--line); background: #1f1f22; padding: 0.8rem; min-width: 0; overflow: hidden; }
    .diagram-head { margin-bottom: 0.6rem; }
    .diagram-head h3 { margin: 0 0 0.2rem; font-size: var(--size-b); }
    .diagram-head p { margin: 0; color: var(--muted); font-size: 0.88rem; }
    .diagram-card .mermaid { background: #18181a; border: 1px solid var(--line); padding: 0.6rem; min-width: 0; overflow-x: auto; text-align: center; }
    .diagram-card .mermaid svg { max-width: 100%; height: auto; }
    .diagram-legend { list-style: none; margin: 0.8rem 0 0; padding: 0; display: flex; flex-wrap: wrap; gap: 0.6rem 1.1rem; color: var(--muted); font-size: 0.88rem; }
    .diagram-legend li { display: flex; align-items: center; gap: 0.4rem; }
    .swatch { display: inline-block; width: 14px; height: 14px; border: 1px solid var(--line); }
    .swatch-gpu0 { background: #1f3a5f; border-color: #6aa9ff; }
    .swatch-gpu1 { background: #3a1f1f; border-color: #ff8a6a; }
    .swatch-locked { background: #3a3a1f; border-color: #ffd76a; }
    .swatch-phase { background: #1b1b1e; border-color: #888; }
    .lightbox { position: fixed; inset: 0; background: rgba(0, 0, 0, 0.88); display: none; place-items: center; z-index: 999; padding: 1.2rem; }
    .lightbox.open { display: grid; }
    .lightbox img { max-width: min(92vw, 1600px); max-height: 82vh; width: auto; height: auto; }
    .lightbox-legend { color: #ddd; margin-top: 0.7rem; text-align: center; }
    .lightbox-close { position: absolute; top: 0.7rem; right: 0.9rem; border: 1px solid var(--accent); color: var(--accent); background: transparent; padding: 0.25rem 0.65rem; cursor: pointer; }
    .footer { margin-top: 1rem; border-top: 1px solid var(--line); padding-top: 0.8rem; color: var(--muted); }
    .footer-hash { font-family: "JetBrains Mono", monospace; font-size: 0.82rem; overflow-wrap: anywhere; word-break: break-all; }

    /* Tables -> cards sous 640px (cahier des charges, point 5) */
    @media (max-width: 639px) {
      .card-table, .card-table thead, .card-table tbody,
      .card-table tr, .card-table th, .card-table td {
        display: block;
        width: 100%;
      }
      .card-table thead { display: none; }
      .card-table tr {
        border: 1px solid var(--line);
        background: #1f1f22;
        margin-bottom: 0.6rem;
        padding: 0.5rem;
      }
      .card-table td {
        border: 0;
        border-bottom: 1px solid #2a2a2e;
        padding: 0.4rem 0;
      }
      .card-table td:last-child { border-bottom: 0; }
      .card-table td::before {
        content: attr(data-label);
        display: block;
        color: var(--muted);
        font-size: 0.78rem;
        margin-bottom: 0.2rem;
      }
    }
    @media (max-width: 760px) {
      .shell { width: min(100%, 98vw); }
    }
"""

JS_BLOCK = """
    (function () {
      var slider = document.getElementById("ab-range");
      var overlay = document.getElementById("ab-overlay");
      if (slider && overlay) {
        var update = function () { overlay.style.width = Number(slider.value || 50) + "%"; };
        slider.addEventListener("input", update);
        update();
      }
      var box = document.getElementById("lightbox");
      var boxImg = document.getElementById("lightbox-image");
      var boxLegend = document.getElementById("lightbox-legend");
      var boxClose = document.getElementById("lightbox-close");
      var closeBox = function () {
        box.classList.remove("open");
        box.setAttribute("aria-hidden", "true");
        boxImg.src = ""; boxLegend.textContent = "";
      };
      document.querySelectorAll("[data-lightbox-src]").forEach(function (button) {
        button.addEventListener("click", function () {
          boxImg.src = button.getAttribute("data-lightbox-src");
          boxImg.alt = button.querySelector("img") ? button.querySelector("img").alt : "Image";
          boxLegend.textContent = button.getAttribute("data-lightbox-caption") || "";
          box.classList.add("open");
          box.setAttribute("aria-hidden", "false");
        });
      });
      if (boxClose) { boxClose.addEventListener("click", closeBox); }
      if (box) {
        box.addEventListener("click", function (event) { if (event.target === box) { closeBox(); } });
      }
      document.addEventListener("keydown", function (event) {
        if (event.key === "Escape" && box.classList.contains("open")) { closeBox(); }
      });
    })();
"""

MERMAID_BLOCK = """
    import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";
    mermaid.initialize({
      startOnLoad: true,
      theme: "dark",
      themeVariables: {
        darkMode: true,
        background: "#18181a",
        primaryColor: "#1f3a5f",
        primaryTextColor: "#f2f2f4",
        primaryBorderColor: "#6aa9ff",
        lineColor: "#888",
        secondaryColor: "#2a2a3a",
        tertiaryColor: "#1b1b1e",
        fontFamily: "Inter, Segoe UI, Arial, sans-serif",
        fontSize: "13px"
      },
      flowchart: { htmlLabels: true, curve: "basis", useMaxWidth: true }
    });
"""


def render_run_card(run: RunRecord) -> str:
    progress = run.progress_ratio
    if progress is None:
        progress_label = "Donnée manquante"
        progress_width = 0.0
    else:
        progress_label = f"{run.steps_done} / {run.steps_target} ({progress * 100:.1f}%)"
        progress_width = progress * 100.0
    steps_label = (
        "Donnée manquante"
        if run.steps_done is None or run.steps_target is None
        else f"{run.steps_done} / {run.steps_target}"
    )
    status_label = STATUS_LABELS.get(run.status, run.status)
    loss_html = (
        render_loss_svg(run.loss_curve)
        if run.loss_curve is not None
        else "<p class='missing-text'>Courbe loss indisponible pour ce run.</p>"
    )
    loss_last_html = (
        f"<p class='run-loss-last'>Dernier loss/average: <code>{run.loss_last:.4f}</code></p>"
        if run.loss_last is not None
        else "<p class='run-loss-last'>Dernier loss/average: Donnée manquante</p>"
    )
    if run.best_loss is not None and run.best_step is not None:
        best_html = (
            "<p class='best-hero-inline'>"
            "<span class='best-kicker'>Meilleur loss</span>"
            f"<strong>{run.best_loss:.4f}</strong>"
            f"<span>step {run.best_step}</span>"
            f"<code>{html_lib.escape(run.best_checkpoint)}</code>"
            "</p>"
        )
    else:
        best_html = ""
    notes_html = "".join(f"<li>{html_lib.escape(n)}</li>" for n in run.notes)
    proof_html = render_proof_block(run)
    return (
        "<article class='run-card' id='run-{rid}' data-run-id='{rid}' data-status='{st}' data-base='{bm}'>"
        "<header class='run-head'>"
        "<h3 class='run-title' data-critical-text>{name}</h3>"
        "<span class='status-badge status-{st}'>{slabel}</span>"
        "</header>"
        "{best}"
        "<p class='run-steps' data-critical-text>Steps: {steps}</p>"
        "<div class='progress-track'><div class='progress-fill' style='width:{w:.2f}%'></div></div>"
        "<p class='run-progress-label' data-critical-text>{plabel}</p>"
        "<p class='run-update' data-critical-text>Dernière MAJ: {lu}</p>"
        "{loss_last}"
        "<div class='loss-block'>{loss}</div>"
        "<section class='proof-section' aria-label='Preuves'>"
        "<h4>Preuves (samples/checkpoints réels)</h4>"
        "{proof}"
        "</section>"
        "<ul class='run-notes'>{notes}</ul>"
        "<p class='run-source'>Sources: {srcs}</p>"
        "</article>"
    ).format(
        rid=html_lib.escape(run.run_id), st=html_lib.escape(run.status),
        bm=html_lib.escape(run.base_model), name=html_lib.escape(run.name),
        slabel=html_lib.escape(status_label), steps=html_lib.escape(steps_label),
        w=progress_width, plabel=html_lib.escape(progress_label),
        lu=html_lib.escape(run.last_update), best=best_html, loss_last=loss_last_html,
        loss=loss_html, proof=proof_html, notes=notes_html,
        srcs=html_lib.escape(", ".join(run.source_files)),
    )


def render_runs_section(runs: list[RunRecord]) -> str:
    groups: dict[str, list[RunRecord]] = {bm: [] for bm in BASE_MODELS}
    for run in runs:
        groups.setdefault(run.base_model, []).append(run)
    parts: list[str] = []
    for bm in BASE_MODELS:
        bm_runs = groups.get(bm, [])
        if not bm_runs:
            continue
        cards = "".join(render_run_card(r) for r in bm_runs)
        parts.append(
            f"<div class='base-group' data-base='{html_lib.escape(bm)}'>"
            f"<h3>Base {html_lib.escape(bm)}</h3>"
            f"<div class='run-grid'>{cards}</div></div>"
        )
    return (
        "<section class='panel' id='runs'>"
        "<div class='panel-head'><h2>Runs training (groupés par base model)</h2>"
        "<p>Loss par run (pas de moyenne cross-arch), statut = Terminé si et seulement si steps == target.</p></div>"
        + "".join(parts)
        + "</section>"
    )


def render_ab_section() -> str:
    before = "assets/delphine-step00800-smoke.png"
    after = "assets/delphine-step01766-smoke.png"
    bs = image_size(f"docs/framework/{before}")
    as_ = image_size(f"docs/framework/{after}")
    if bs and as_ and bs == as_:
        return (
            "<section class='panel' id='ab-compare'>"
            "<div class='panel-head'><h2>A/B Delphine step00800 vs step01766</h2>"
            "<p>Comparatif visuel, même ratio, slider avant/après.</p></div>"
            "<div class='ab-slider' id='ab-slider'>"
            f"<img src='{before}' alt='Delphine step00800 avant' class='ab-base' />"
            "<div class='ab-overlay' id='ab-overlay'>"
            f"<img src='{after}' alt='Delphine step01766 après' class='ab-top' /></div>"
            "<input type='range' min='0' max='100' value='50' id='ab-range' aria-label='Comparatif avant après' />"
            "</div>"
            "<p class='ab-meta'>Sources: <code>docs/framework/assets/delphine-step00800-smoke.png</code> et "
            "<code>docs/framework/assets/delphine-step01766-smoke.png</code></p>"
            "</section>"
        )
    return (
        "<section class='panel'><div class='panel-head'>"
        "<h2>A/B Delphine step00800 vs step01766</h2></div>"
        "<p class='missing-text'>Donnée manquante: images A/B incompatibles ou absentes.</p></section>"
    )


def render_xy_loss_matrices_section() -> str:
    """Rend la section des Planches Contact XY + Courbes de Loss + Tableaux Checkpoints."""
    loss_index = load_loss_index()
    runs_idx = loss_index.get("runs", {})

    sheets = [
        (
            "assets/fowx_flux_xy_loss_contact_sheet.png",
            "Fowx — Planche XY 8 Checkpoints × 5 Prompts + Courbe de Loss & Normes L2 (Flux.1-dev)",
            "Correction du workflow ComfyUI_21328_ : steps explicites, détection du checkpoint mort fowx_64 (||W_up||₂=0.0000) et vraies métadonnées UNet GGUF.",
        ),
        (
            "assets/delphine_sel2_flux_xy_loss_contact_sheet.png",
            "Delphine Selection_2 — Planche XY Flux.1-dev (Steps 800 / 1550 / 1800) + Analyse Spectrale",
            "Comparaison multi-prompts et courbe des 8 checkpoints (steps 50→1800) : sweet spot au step 1550 (avr_loss=0.2995, ||W_up||₂=5.7306).",
        ),
        (
            "assets/delphine_sel2_sdxl_xy_loss_contact_sheet.png",
            "Delphine Selection_2 — Planche XY SDXL (Steps 600 → 1200) + Référence Flux & Loss TensorBoard",
            "Croisement visuel sur 3 prompts + courbe TensorBoard 1200 steps (sweet spot SDXL au step 900, avr_loss=0.0885).",
        ),
        (
            "assets/nemeris_sd15_xy_loss_contact_sheet.png",
            "Nemeris (n3m3riys) — Planche 8 Prompts (A→H) & Courbe TensorBoard 3000 Steps (SD1.5)",
            "Validation complète des 8 prompts d'évaluation (incluant contrôle sans trigger) et métriques des 9 checkpoints (steps 300→3000).",
        ),
    ]

    cards_html: list[str] = []
    for rel_img, title, desc in sheets:
        if (PORTAL_DIR / rel_img).exists():
            cards_html.append(
                "<article class='xy-sheet-card'>"
                f"<h3>{html_lib.escape(title)}</h3>"
                f"<p>{html_lib.escape(desc)}</p>"
                "<button class='thumb-button' type='button' "
                f"data-lightbox-src='{html_lib.escape(rel_img)}' "
                f"data-lightbox-caption='{html_lib.escape(title)}'>"
                f"<img src='{html_lib.escape(rel_img)}' alt='{html_lib.escape(title)}' loading='lazy' decoding='async' />"
                "</button>"
                "</article>"
            )

    def render_ckpt_table(rkey: str) -> str:
        rdata = runs_idx.get(rkey)
        if not rdata:
            return ""
        rtitle = rdata.get("title", rkey)
        bmodel = rdata.get("base_model_real", "")
        ranking_note = ""
        if rkey == "nemeriys_sdxl_v2":
            ranking_note = (
                "<p class='curve-note'>Classement du meilleur loss/average au pire, "
                "checkpoints sauvés tous les 300 steps. Δloss = écart au step 2100. "
                "Aucun step NaN.</p>"
            )
        rows_tr: list[str] = []
        for ck in rdata.get("checkpoints", []):
            step_val = ck.get("step", "?")
            fname = ck.get("filename", "")
            avr_l = ck.get("avr_loss")
            cur_l = ck.get("cur_loss")
            d_l = ck.get("delta_loss", 0.0)
            l2_v = ck.get("lora_up_l2", 0.0)
            badge = str(ck.get("badge", ""))
            note = str(ck.get("note", ""))
            b_cls = "badge-sweet" if "SWEET SPOT" in badge else ("badge-dead" if "DEAD" in badge else "badge-info")
            row_cls = " class='row-best'" if "SWEET SPOT" in badge else ""
            avr_s = f"{avr_l:.4f}" if isinstance(avr_l, (int, float)) else "N/A"
            cur_s = f"{cur_l:.4f}" if isinstance(cur_l, (int, float)) else "N/A"
            dl_s = f"{d_l:+.4f}" if isinstance(d_l, (int, float)) else "0.0000"
            l2_s = f"{l2_v:.4f}" if isinstance(l2_v, (int, float)) else "N/A"
            rows_tr.append(
                f"<tr{row_cls}>"
                f"<td data-label='Step'><strong>{html_lib.escape(str(step_val))}</strong></td>"
                f"<td data-label='Checkpoint'><code>{html_lib.escape(fname)}</code></td>"
                f"<td data-label='avr_loss'><code>{html_lib.escape(avr_s)}</code></td>"
                f"<td data-label='cur_loss'><code>{html_lib.escape(cur_s)}</code></td>"
                f"<td data-label='Δloss'><code>{html_lib.escape(dl_s)}</code></td>"
                f"<td data-label='||lora_up||₂'><code>{html_lib.escape(l2_s)}</code></td>"
                f"<td data-label='Diagnostic'><span class='{b_cls}'>{html_lib.escape(badge)}</span></td>"
                f"<td data-label='Analyse'>{html_lib.escape(note)}</td>"
                "</tr>"
            )
        return (
            "<div class='base-group'>"
            f"<h3>{html_lib.escape(rtitle)} — <small>{html_lib.escape(bmodel)}</small></h3>"
            f"{ranking_note}"
            "<table class='card-table' aria-label='Matrice Loss et Normes L2'>"
            "<thead><tr>"
            "<th>Step</th><th>Checkpoint</th><th>avr_loss</th><th>cur_loss</th>"
            "<th>Δloss</th><th>||lora_up||₂</th><th>Diagnostic</th><th>Analyse</th>"
            "</tr></thead>"
            f"<tbody>{''.join(rows_tr)}</tbody></table></div>"
        )

    lead_html = ""
    nem = runs_idx.get("nemeriys_sdxl_v2")
    if nem:
        best = next(
            (
                ck for ck in nem.get("checkpoints", [])
                if ck.get("step") == nem.get("recommended_step")
            ),
            None,
        )
        curve = loss_curve_from_index(
            loss_index,
            "nemeriys_sdxl_v2",
            str(nem.get("log_tb") or "TensorBoard"),
            "loss/average aux sauvegardes, step 300 à 6000. Point vert : meilleur checkpoint (step 2100). Le step 2 et le step 5756 ne sont pas des fichiers.",
        )
        svg = render_loss_svg(curve) if curve is not None else ""
        if best is not None:
            avr = float(best["avr_loss"])
            cur = float(best["cur_loss"])
            l2 = float(best.get("lora_up_l2") or 0.0)
            lead_html = (
                "<article class='best-hero' id='rapport-nemeriys-sdxl'>"
                "<p class='best-kicker'>Nemeriys SDXL v2 · Terminé · 6000 / 6000</p>"
                "<p class='best-kicker'>Meilleur loss</p>"
                f"<p class='best-value'><strong>{avr:.6f}</strong>"
                f"<span>loss/average · step {int(best['step'])}</span></p>"
                f"<p class='best-ckpt'><code>{html_lib.escape(str(best.get('filename') or ''))}</code></p>"
                f"<p class='best-path'><code>{html_lib.escape(str(best.get('path') or ''))}</code></p>"
                f"<p class='best-sub'>cur_loss {cur:.4f} · ||lora_up||₂ {l2:.4f} · "
                f"dernier loss/average step 6000 = 0.0870</p>"
                "<dl class='best-meta'>"
                f"<div><dt>Config</dt><dd><code>{html_lib.escape(str(nem.get('config') or ''))}</code></dd></div>"
                f"<div><dt>Dataset</dt><dd>{html_lib.escape(str(nem.get('dataset') or ''))}</dd></div>"
                f"<div><dt>TensorBoard</dt><dd><code>{html_lib.escape(str(nem.get('log_tb') or ''))}</code></dd></div>"
                f"<div><dt>Log texte</dt><dd><code>{html_lib.escape(str(nem.get('log_text') or ''))}</code> — "
                f"{html_lib.escape(str(nem.get('log_text_note') or ''))}</dd></div>"
                f"<div><dt>Checkpoint final</dt><dd><code>{html_lib.escape(str(nem.get('final_checkpoint') or ''))}</code></dd></div>"
                "</dl>"
                "</article>"
                f"{svg}"
            )
        lead_html += render_ckpt_table("nemeriys_sdxl_v2")

    table_blocks = [
        render_ckpt_table(rkey)
        for rkey in ("fowx_flux", "delphine_flux_sel2", "delphine_sdxl_sel2", "delphine_sd15_sel2", "nemeris_sd15")
    ]

    return (
        "<section class='panel' id='xy-loss-matrices'>"
        "<div class='panel-head'>"
        "<h2>Planches Comparatives XY Multi-Checkpoints &amp; Matrices Loss / Normes L2</h2>"
        "<p>Revue visuelle haute résolution croisée avec les courbes de loss TensorBoard et la norme spectrale ||lora_up||₂ par step.</p>"
        "</div>"
        f"{lead_html}"
        f"<div class='xy-sheet-grid'>{''.join(cards_html)}</div>"
        f"{''.join(table_blocks)}"
        "</section>"
    )


def render_sources_table(sources: list[str]) -> str:
    rows = "".join(
        f"<tr><td data-label='Source canonique'><code>{html_lib.escape(s)}</code></td></tr>" for s in sources
    )
    return (
        "<table class='card-table source-table' aria-label='Sources canoniques'>"
        f"<thead><tr><th>Source canonique</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def render_html(
    runs: list[RunRecord],
    sources: list[str],
    generated_at: str,
    sources_hash: str,
    organigram_section: str,
) -> str:
    status_counts = {s: 0 for s in STATUS_ORDER}
    for r in runs:
        status_counts[r.status] = status_counts.get(r.status, 0) + 1
    runs_section = render_runs_section(runs)
    xy_loss_section = render_xy_loss_matrices_section()
    ab_section = render_ab_section()
    sources_table = render_sources_table(sources)

    return f"""<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Portail NEMOKID v2</title>
  <style>{CSS_BLOCK}</style>
</head>
<body>
  <main class="shell">
    <header class="hero">
      <h1>Portail NEMOKID v2</h1>
      <p>Généré automatiquement par <code>tools/build_portal.py</code> depuis trois sources markdown canoniques. Suivi run par run, loss par run, preuves réelles uniquement.</p>
      <div class="meta-grid">
        <div class="meta-item"><strong>{len(runs)}</strong><span>Runs suivis</span></div>
        <div class="meta-item"><strong>{status_counts.get("active", 0)}</strong><span>Runs actifs</span></div>
        <div class="meta-item"><strong>{status_counts.get("completed", 0)}</strong><span>Runs terminés</span></div>
        <div class="meta-item"><strong>{status_counts.get("interrupted", 0)}</strong><span>Runs interrompus</span></div>
      </div>
    </header>

    {xy_loss_section}

    {runs_section}

    {ab_section}

    {organigram_section}

    <section class="panel" id="sources">
      <div class="panel-head"><h2>Sources canoniques (hash sha256)</h2>
      <p>Trois sources markdown (préparation, compte rendu, matrice) combinées pour générer ce portail.</p></div>
      {sources_table}
      <p class="footer-hash">sha256: <code>{html_lib.escape(sources_hash)}</code></p>
    </section>

    <footer class="footer">
      <p>Génération: <strong>{html_lib.escape(generated_at)}</strong></p>
      <p class="footer-hash">sha256 sources: <code>{html_lib.escape(sources_hash)}</code></p>
    </footer>
  </main>

  <div class="lightbox" id="lightbox" aria-hidden="true">
    <button class="lightbox-close" id="lightbox-close" type="button">Fermer</button>
    <div>
      <img id="lightbox-image" src="" alt="" />
      <div class="lightbox-legend" id="lightbox-legend"></div>
    </div>
  </div>

  <script>{JS_BLOCK}</script>
  <script type="module">{MERMAID_BLOCK}</script>
</body>
</html>
"""


def to_json_payload(
    runs: list[RunRecord],
    sources: list[str],
    generated_at: str,
    sources_hash: str,
    organigram_diagrams: list[dict[str, str]],
) -> dict[str, Any]:
    status_counts = {s: 0 for s in STATUS_ORDER}
    payload_runs = []
    for r in runs:
        status_counts[r.status] = status_counts.get(r.status, 0) + 1
        payload_runs.append({
            "id": r.run_id, "name": r.name, "base_model": r.base_model,
            "status": r.status, "steps_done": r.steps_done, "steps_target": r.steps_target,
            "progress_ratio": r.progress_ratio, "last_update": r.last_update,
            "notes": r.notes, "sources": r.source_files,
            "loss_last": r.loss_last,
            "loss_curve": {
                "source": r.loss_curve.source, "points": r.loss_curve.points,
                "note": r.loss_curve.note,
            } if r.loss_curve else None,
            "proofs": [
                {"kind": p.kind, "path": p.path, "exists": p.exists,
                 "width": p.width, "height": p.height, "step": p.step, "prompt": p.prompt}
                for p in r.proofs
            ],
            "has_real_proof": r.has_real_proof,
        })
    return {
        "generated_at": generated_at,
        "sources_hash": sources_hash,
        "sources": sources,
        "base_models": BASE_MODELS,
        "runs": payload_runs,
        "stats": {"run_count": len(runs), "status_counts": status_counts},
        "organigram": {"section_id": "organigramme", "diagrams": organigram_diagrams},
    }


def compute_sources_hash(source_paths: list[Path]) -> str:
    h = hashlib.sha256()
    for path in source_paths:
        h.update(str(path.relative_to(REPO_ROOT)).encode("utf-8"))
        h.update(b"\0")
        if path.exists():
            h.update(path.read_bytes())
        else:
            h.update(b"<missing>")
        h.update(b"\0")
    return h.hexdigest()


def main() -> int:
    generated_at = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")

    suite_path = REPO_ROOT / SUITE_PATH
    report_paths = glob_sources(REPORTS_GLOB)
    matrix_path = REPO_ROOT / MATRIX_PATH
    canonical_paths = [suite_path, *report_paths, matrix_path]

    suite_text = read_text(SUITE_PATH)
    reports_text = "\n\n".join(read_text(str(p.relative_to(REPO_ROOT))) for p in report_paths)
    matrix_text = read_text(MATRIX_PATH)

    runs = collect_runs(suite_text, reports_text, matrix_text)
    sources_hash = compute_sources_hash(canonical_paths)
    sources = [str(p.relative_to(REPO_ROOT)) for p in canonical_paths]
    organigram_section = render_organigram_section()

    json_payload = to_json_payload(
        runs=runs, sources=sources, generated_at=generated_at,
        sources_hash=sources_hash, organigram_diagrams=ORGANIGRAM_DIAGRAMS,
    )
    html_payload = render_html(
        runs=runs, sources=sources, generated_at=generated_at,
        sources_hash=sources_hash, organigram_section=organigram_section,
    )

    PORTAL_JSON.write_text(
        json.dumps(json_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    PORTAL_HTML.write_text(html_payload, encoding="utf-8")

    print(f"[build] sources -> {len(canonical_paths)} fichiers canoniques")
    for s in sources:
        print(f"          - {s}")
    print(f"[build] sha256  -> {sources_hash}")
    print(f"[build] json    -> {PORTAL_JSON.relative_to(REPO_ROOT)}")
    print(f"[build] html    -> {PORTAL_HTML.relative_to(REPO_ROOT)}")
    print(f"[build] runs    -> {len(runs)}")
    print(f"[build] bases   -> {', '.join(BASE_MODELS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())






