"""Generate figures for the ML.ENERGY Leaderboard v3.0 blog post.

Each section function generates one or two SVG/PNG figures and prints
key statistics cited in the post. Demonstrates the mlenergy-data toolkit's
typed collection API (LLMRuns, DiffusionRuns) for loading, filtering,
grouping, and analyzing benchmark data.

Usage:
    # From compiled data directory (parquet-first, fast)
    uv run --with matplotlib --with numpy python blog_analysis_scripts.py \
        --mlenergy-data-dir /path/to/compiled/data

    # From raw benchmark results
    uv run --with matplotlib --with numpy python blog_analysis_scripts.py \
        --results-dir /path/to/llm/h100/current/run \
        --results-dir /path/to/llm/b200/current/run \
        --results-dir /path/to/diffusion/h100/current/run \
        --results-dir /path/to/diffusion/b200/current/run

    # From Hugging Face Hub (default)
    uv run --with matplotlib --with numpy python blog_analysis_scripts.py
"""

from __future__ import annotations

import re
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from mlenergy_data.records.runs import DiffusionRun, DiffusionRuns, LLMRun, LLMRuns

mpl.rcParams["svg.hashsalt"] = "42"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)


@dataclass
class FP8BF16Comparison:
    task: str
    model: str
    gpu: str
    num_gpus: int
    batch: int
    fp8_energy: float
    bf16_energy: float
    energy_ratio: float  # <1 means FP8 wins
    fp8_latency: float
    bf16_latency: float
    latency_ratio: float  # <1 means FP8 wins (faster)


@dataclass
class IsoTimeComparison:
    model: str
    task: str
    deadline: float
    h100_energy: float
    b200_energy: float
    gain_pct: float  # >0 means B200 wins


FIGURES_DIR = Path("figures")
MIN_BATCH_SIZE = 16

MLENERGY_GREEN = "#23d175"
MLENERGY_BG = "#2e303e"

COLORS = {
    "green": MLENERGY_GREEN,
    "red": "#e74c3c",
    "blue": "#3498db",
    "orange": "#f39c12",
    "purple": "#9b59b6",
    "gray": "gray",
}

FONT_PARAMS = {
    "font.size": 10,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
}


LLM_RUNS: LLMRuns | None = None
DIFF_RUNS: DiffusionRuns | None = None


def setup(
    mlenergy_data_dir: str | Path | None = None,
    results_dirs: list[str | Path] | None = None,
    llm_config_dir: str | Path | None = None,
    diffusion_config_dir: str | Path | None = None,
) -> None:
    """Load LLM and Diffusion benchmark data.

    Three loading modes:
      1. ``--mlenergy-data-dir``: compiled data directory with parquet files.
      2. ``--results-dir``: one or more raw benchmark result directories.
      3. Neither: load from Hugging Face Hub.
    """
    global LLM_RUNS, DIFF_RUNS
    if mlenergy_data_dir is not None:
        LLM_RUNS = LLMRuns.from_directory(mlenergy_data_dir)
        DIFF_RUNS = DiffusionRuns.from_directory(mlenergy_data_dir)
    elif results_dirs:
        LLM_RUNS = LLMRuns.from_raw_results(*results_dirs, config_dir=llm_config_dir)
        DIFF_RUNS = DiffusionRuns.from_raw_results(*results_dirs, config_dir=diffusion_config_dir)
    else:
        LLM_RUNS = LLMRuns.from_hf()
        DIFF_RUNS = DiffusionRuns.from_hf()
    llm_tasks = sorted(set(r.task for r in LLM_RUNS))
    diff_tasks = sorted(set(r.task for r in DIFF_RUNS))
    print(f"Loaded {len(LLM_RUNS)} LLM runs ({llm_tasks}), "
          f"{len(DIFF_RUNS)} diffusion runs ({diff_tasks})")


def _llm() -> LLMRuns:
    if LLM_RUNS is None:
        raise RuntimeError("Call setup() before running analyses")
    return LLM_RUNS


def _diff() -> DiffusionRuns:
    if DIFF_RUNS is None:
        raise RuntimeError("Call setup() before running analyses")
    return DIFF_RUNS


def format_model_with_params(model: str, params: float) -> str:
    """Add params to model name if no size marker (e.g. '8B') is present."""
    if re.search(r'\d+\.?\d*B\b', model):
        return model
    return f"{model} ({params:.1f}B)"


def print_header(title: str, width: int = 80) -> None:
    print("=" * width)
    print(title)
    print("=" * width)


def savefig(fig_fn, prefix: str) -> None:
    """Save figure as SVG and PNG in light and dark themes."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for theme in ["light", "dark"]:
        mpl_style = "default" if theme == "light" else "dark_background"

        with plt.style.context(mpl_style):
            mpl.rcParams["svg.hashsalt"] = "42"
            with plt.rc_context(FONT_PARAMS):
                fig = fig_fn()

                saved_paths = []
                for fmt in ["svg", "png"]:
                    path = FIGURES_DIR / f"{prefix}-{theme}.{fmt}"
                    if fmt == "svg":
                        fig.savefig(path, metadata={"Date": None}, transparent=True, bbox_inches="tight")
                    else:
                        fig.savefig(path, dpi=150, transparent=True, bbox_inches="tight")
                    saved_paths.append(str(path))

                plt.close(fig)
                print(f"Saved: {', '.join(saved_paths)}")


def section1_1_llm_tokens() -> None:
    """LLM energy by task: output length drives energy per response."""
    print_header("SECTION 1.1: More Tokens, More Energy (LLMs)")

    gpqa_runs = _llm().task("gpqa")
    chat_runs = _llm().task("lm-arena-chat")

    def get_min_energy_runs(runs: LLMRuns) -> dict[str, LLMRun]:
        """Per-model minimum-energy configuration on B200."""
        return {
            nick: min(group, key=lambda r: r.energy_per_token_joules)
            for nick, group in runs.gpu("B200").group_by("nickname").items()
        }

    gpqa_by_model = get_min_energy_runs(gpqa_runs)
    chat_by_model = get_min_energy_runs(chat_runs)

    gpqa_tokens_mean = float(np.mean([r.avg_output_len for r in gpqa_by_model.values()]))
    chat_tokens_mean = float(np.mean([r.avg_output_len for r in chat_by_model.values()]))
    gpqa_energy_mean = float(np.mean([r.energy_per_request_joules for r in gpqa_by_model.values()]))
    chat_energy_mean = float(np.mean([r.energy_per_request_joules for r in chat_by_model.values()]))

    print(f"Output tokens: Problem Solving {gpqa_tokens_mean:,.0f} vs Text Conversation {chat_tokens_mean:,.0f} = {gpqa_tokens_mean / chat_tokens_mean:.0f}x")
    print(f"Energy/response: {gpqa_energy_mean:,.0f} J vs {chat_energy_mean:,.0f} J = {gpqa_energy_mean / chat_energy_mean:.0f}x")

    # Case study: Qwen 3 32B on 1x B200 (blog Table 1)
    model = "Qwen 3 32B"
    gpqa_32b = list(gpqa_runs.nickname(model).gpu("B200").num_gpus(1))
    chat_32b = list(chat_runs.nickname(model).gpu("B200").num_gpus(1))
    gpqa_max = max(gpqa_32b, key=lambda r: r.max_num_seqs)
    chat_max = max(chat_32b, key=lambda r: r.max_num_seqs)
    same_batch = gpqa_max.max_num_seqs
    chat_at_batch = next((r for r in chat_32b if r.max_num_seqs == same_batch), None)

    print(f"\nCase study: {model} on 1x B200")
    print(f"  Max batch: {chat_max.max_num_seqs} (chat) vs {gpqa_max.max_num_seqs} (gpqa)")
    print(f"  Output tokens: {chat_max.avg_output_len:.0f} vs {gpqa_max.avg_output_len:.0f}")
    print(f"  E/token @ max batch: {chat_max.energy_per_token_joules:.3f} vs {gpqa_max.energy_per_token_joules:.3f} J")
    if chat_at_batch:
        print(f"  E/token @ batch {same_batch}: {chat_at_batch.energy_per_token_joules:.3f} vs {gpqa_max.energy_per_token_joules:.3f} J")
    print(f"  E/response: {chat_max.energy_per_request_joules:,.0f} vs {gpqa_max.energy_per_request_joules:,.0f} J")

    def draw():
        fig, axes = plt.subplots(1, 3, figsize=(14, 5), tight_layout=True)

        gpqa_best = list(get_min_energy_runs(gpqa_runs).values())
        chat_best = list(get_min_energy_runs(chat_runs).values())

        gpqa_energy_tok = [r.energy_per_token_joules for r in gpqa_best]
        chat_energy_tok = [r.energy_per_token_joules for r in chat_best]
        gpqa_tokens = [r.avg_output_len for r in gpqa_best]
        chat_tokens = [r.avg_output_len for r in chat_best]
        gpqa_energy = [r.energy_per_request_joules for r in gpqa_best]
        chat_energy = [r.energy_per_request_joules for r in chat_best]

        positions = [1, 1.7]
        width = 0.6

        def style_violin(parts):
            parts["bodies"][0].set_facecolor(COLORS["red"])
            parts["bodies"][1].set_facecolor(COLORS["blue"])
            for pc in parts["bodies"]:
                pc.set_zorder(3)

        ax1 = axes[0]
        parts1 = ax1.violinplot([gpqa_tokens, chat_tokens], positions=positions, widths=width, showmedians=True)
        style_violin(parts1)
        ax1.set_xticks(positions)
        ax1.set_xticklabels(["Problem\nSolving", "Text\nConversation"], fontsize=12)
        ax1.set_ylabel("Output Tokens", fontsize=13)
        ax1.set_title("Output Length", fontsize=15)
        ax1.tick_params(axis='y', labelsize=11)
        ax1.set_xlim(0.5, 2.2)
        ax1.set_ylim(0, None)

        ax2 = axes[1]
        parts2 = ax2.violinplot([gpqa_energy_tok, chat_energy_tok], positions=positions, widths=width, showmedians=True)
        style_violin(parts2)
        ax2.set_xticks(positions)
        ax2.set_xticklabels(["Problem\nSolving", "Text\nConversation"], fontsize=12)
        ax2.set_ylabel("Energy per Token (J)", fontsize=13)
        ax2.set_title("Energy per Token", fontsize=15)
        ax2.tick_params(axis='y', labelsize=11)
        ax2.set_xlim(0.5, 2.2)
        ax2.set_ylim(0, None)

        ax3 = axes[2]
        parts3 = ax3.violinplot([gpqa_energy, chat_energy], positions=positions, widths=width, showmedians=True)
        style_violin(parts3)
        ax3.set_xticks(positions)
        ax3.set_xticklabels(["Problem\nSolving", "Text\nConversation"], fontsize=12)
        ax3.set_ylabel("Energy per Response (J)", fontsize=13)
        ax3.set_title("Energy per Response", fontsize=15)
        ax3.tick_params(axis='y', labelsize=11)
        ax3.set_xlim(0.5, 2.2)
        ax3.set_ylim(0, None)

        return fig

    savefig(draw, "section1-1-llm")


def section1_2_mllm() -> None:
    """Multimodal LLMs: image and video inputs increase energy."""
    print_header("SECTION 1.2: Multimodal LLMs")

    text_chat = _llm().task("lm-arena-chat")
    image_chat = _llm().task("image-chat")
    video_chat = _llm().task("video-chat")

    vlm_to_text_map = {
        "Qwen 3 VL 8B Instruct": "Qwen 3 8B",
        "Qwen 3 VL 32B Instruct": "Qwen 3 32B",
        "Qwen 3 VL 30B A3B Instruct": "Qwen 3 30B A3B Instruct",
        "Qwen 3 Omni 30B A3B Instruct": "Qwen 3 30B A3B Instruct",
        "Qwen 3 VL 235B A22B Instruct": "Qwen 3 235B A22B Instruct",
        "Qwen 3 VL 235B A22B Instruct FP8": "Qwen 3 235B A22B Instruct FP8",
    }

    def get_min_energy_run(runs: LLMRuns, model: str, num_gpus: int | None = None) -> LLMRun | None:
        filtered = runs.nickname(model).gpu("B200")
        if num_gpus is not None:
            filtered = filtered.num_gpus(num_gpus)
        if not filtered:
            filtered = runs.nickname(model)
            if num_gpus is not None:
                filtered = filtered.num_gpus(num_gpus)
        return min(filtered, key=lambda r: r.energy_per_token_joules) if filtered else None

    def get_common_gpu_count(text_runs: LLMRuns, img_runs: LLMRuns, vid_runs: LLMRuns,
                              text_model: str, vlm_model: str) -> int | None:
        """Find the GPU count where all three modalities have B200 runs."""
        txt_gpus = {r.num_gpus for r in text_runs.nickname(text_model).gpu("B200")}
        img_gpus = {r.num_gpus for r in img_runs.nickname(vlm_model).gpu("B200")}
        vid_gpus = {r.num_gpus for r in vid_runs.nickname(vlm_model).gpu("B200")}
        common = txt_gpus & img_gpus & vid_gpus
        if not common:
            return None
        best_gpu = None
        best_energy = float("inf")
        for gpu in common:
            run = get_min_energy_run(text_runs, text_model, gpu)
            if run and run.energy_per_token_joules < best_energy:
                best_energy = run.energy_per_token_joules
                best_gpu = gpu
        return best_gpu

    # Compute energy ratios across all VLM models (blog: "1.1-5.2x image, 1.3-15.0x video")
    image_ratios: list[float] = []
    video_ratios: list[float] = []
    for vlm, text_model in vlm_to_text_map.items():
        common_gpus = get_common_gpu_count(text_chat, image_chat, video_chat, text_model, vlm)
        if common_gpus is None:
            continue
        txt_run = get_min_energy_run(text_chat, text_model, common_gpus)
        img_run = get_min_energy_run(image_chat, vlm, common_gpus)
        vid_run = get_min_energy_run(video_chat, vlm, common_gpus)
        if txt_run and img_run:
            image_ratios.append(img_run.energy_per_token_joules / txt_run.energy_per_token_joules)
        if txt_run and vid_run:
            video_ratios.append(vid_run.energy_per_token_joules / txt_run.energy_per_token_joules)

    print(f"Image vs text energy/token: {min(image_ratios):.1f}-{max(image_ratios):.1f}x")
    print(f"Video vs text energy/token: {min(video_ratios):.1f}-{max(video_ratios):.1f}x")

    # Average output lengths (blog footnote)
    text_models = set(vlm_to_text_map.values())
    vlm_models = set(vlm_to_text_map.keys())
    text_lens = text_chat.nickname(*text_models).data.avg_output_len
    image_lens = image_chat.nickname(*vlm_models).data.avg_output_len
    video_lens = video_chat.nickname(*vlm_models).data.avg_output_len
    print(f"Avg output lengths: Text {sum(text_lens)/len(text_lens):.0f}, "
          f"Image {sum(image_lens)/len(image_lens):.0f}, "
          f"Video {sum(video_lens)/len(video_lens):.0f}")

    def draw():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.set_axisbelow(True)
        ax2.set_axisbelow(True)

        plot_models = [
            ("Qwen 3 VL 8B Instruct", "Qwen 3 8B"),
            ("Qwen 3 VL 30B A3B Instruct", "Qwen 3 30B A3B Instruct"),
            ("Qwen 3 VL 235B A22B Instruct", "Qwen 3 235B A22B Instruct"),
        ]

        model_labels = []
        text_energies = []
        image_energies = []
        video_energies = []

        for vlm, txt_model in plot_models:
            common_gpus = get_common_gpu_count(text_chat, image_chat, video_chat, txt_model, vlm)
            if common_gpus is None:
                continue
            txt_r = get_min_energy_run(text_chat, txt_model, common_gpus)
            img_r = get_min_energy_run(image_chat, vlm, common_gpus)
            vid_r = get_min_energy_run(video_chat, vlm, common_gpus)

            if txt_r and img_r:
                short_name = vlm.replace("Qwen 3 VL ", "").replace(" Instruct", "")
                model_labels.append(short_name)
                text_energies.append(txt_r.energy_per_token_joules)
                image_energies.append(img_r.energy_per_token_joules)
                video_energies.append(vid_r.energy_per_token_joules if vid_r else 0)

        x = np.arange(len(model_labels))
        width = 0.25

        ax1.grid(True, axis="y", alpha=0.3)
        ax1.bar(x - width, text_energies, width, label="Text", color=COLORS["blue"])
        ax1.bar(x, image_energies, width, label="Image", color=COLORS["red"])
        ax1.bar(x + width, video_energies, width, label="Video", color=COLORS["green"])

        vlm_model = "Qwen 3 VL 8B Instruct"
        text_model = "Qwen 3 8B"

        common_gpus = get_common_gpu_count(text_chat, image_chat, video_chat, text_model, vlm_model)
        txt_r = get_min_energy_run(text_chat, text_model, common_gpus)
        img_r = get_min_energy_run(image_chat, vlm_model, common_gpus)
        vid_r = get_min_energy_run(video_chat, vlm_model, common_gpus)

        modalities = ["Text", "Image", "Video"]
        batch_sizes = [txt_r.max_num_seqs, img_r.max_num_seqs, vid_r.max_num_seqs]
        energy_per_tok = [txt_r.energy_per_token_joules, img_r.energy_per_token_joules, vid_r.energy_per_token_joules]
        # From prometheus.json steady_state_stats."vllm:kv_cache_usage_perc"
        kv_cache_util = [84.5, 67.6, 10.4]  # %

        energy_y_max = max(max(image_energies), max(video_energies), max(energy_per_tok)) * 1.15

        ax1.set_ylabel("Energy per Token (J)", fontsize=13)
        ax1.set_title("Qwen 3 (VL) Family Energy per Token", fontsize=12)
        ax1.set_xticks(x)
        ax1.set_xticklabels(model_labels, fontsize=11)
        ax1.tick_params(axis="y", labelsize=11)
        ax1.legend(fontsize=10)
        ax1.set_ylim(0, energy_y_max)

        x2 = np.arange(len(modalities))

        ax2.grid(True, alpha=0.3)
        ax2_twin = ax2.twinx()

        line1, = ax2.plot(x2, batch_sizes, "o-", color=COLORS["blue"], linewidth=2, markersize=8, label="Batch Size")
        ax2.set_ylabel("Batch Size", fontsize=13, color=COLORS["blue"])
        ax2.tick_params(axis="y", labelsize=11, colors=COLORS["blue"])
        ax2.set_ylim(0, max(batch_sizes) * 1.15)

        line2, = ax2_twin.plot(x2, kv_cache_util, "^-", color=COLORS["red"], linewidth=2, markersize=8, label="KV Cache Utilization")
        ax2_twin.set_ylabel("KV Cache Utilization (%)", fontsize=13, color=COLORS["red"])
        ax2_twin.tick_params(axis="y", labelsize=11, colors=COLORS["red"])
        ax2_twin.set_ylim(0, 100)

        ax2.set_title("Qwen 3 (VL) 8B Metrics on 1x B200", fontsize=12)
        ax2.set_xticks(x2)
        ax2.set_xticklabels(modalities, fontsize=12)
        ax2.tick_params(axis="x", labelsize=11)
        ax2.legend(handles=[line1, line2], fontsize=10, loc="upper right")

        fig.tight_layout()
        return fig

    savefig(draw, "section1-2-mllm")


def section1_3_diffusion() -> None:
    """Diffusion models: energy varies by steps, resolution, and frames."""
    print_header("SECTION 1.3: Diffusion Models")

    text_to_image = _analyze_text_to_image()
    text_to_video = _analyze_text_to_video()

    def draw():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.set_axisbelow(True)
        ax2.set_axisbelow(True)

        t2i = text_to_image[:7]
        models1 = [format_model_with_params(r.nickname, r.total_params_billions).replace("Stable Diffusion", "SD") for r in t2i]
        energies1 = [r.energy_per_generation_joules / 1000 for r in t2i]
        ax1.grid(True, axis='y', alpha=0.3)
        bars1 = ax1.bar(models1, energies1, color=COLORS["green"])
        ax1.set_ylabel("Energy per Image (kJ)", fontsize=13)
        ax1.set_title("Text-to-Image", fontsize=12)
        ax1.set_ylim(0, max(energies1) * 1.15)
        ax1.tick_params(axis='y', labelsize=11)
        plt.sca(ax1)
        plt.xticks(rotation=35, ha='right', fontsize=11)

        hunyuan_idx = next(i for i, r in enumerate(t2i) if "Hunyuan" in r.nickname)
        ax1.annotate(
            "1.5B model > 8.1B model\n(50 vs 28 steps)",
            xy=(hunyuan_idx, t2i[hunyuan_idx].energy_per_generation_joules / 1000),
            xytext=(hunyuan_idx - 2.0, t2i[hunyuan_idx].energy_per_generation_joules / 1000 + 0.8),
            fontsize=11,
            ha='center',
            arrowprops=dict(arrowstyle='->', color='gray', lw=1),
        )

        t2v = text_to_video
        models2 = [format_model_with_params(r.nickname, r.total_params_billions) for r in t2v]
        energies2 = [r.energy_per_generation_joules / 1000 for r in t2v]
        ax2.grid(True, axis='y', alpha=0.3)
        bars2 = ax2.bar(models2, energies2, color=COLORS["red"])
        ax2.set_ylabel("Energy per Video (kJ)", fontsize=13)
        ax2.set_title("Text-to-Video", fontsize=12)
        ax2.set_ylim(0, max(energies2) * 1.15)
        ax2.tick_params(axis='y', labelsize=11)
        plt.sca(ax2)
        plt.xticks(rotation=35, ha='right', fontsize=11)

        for i in range(2):
            ax2.text(i, energies2[i] + 50, f"{energies2[i]:.0f}", ha='center', fontsize=10)

        cog_idx = next(i for i, r in enumerate(t2v) if "CogVideoX 1.5" in r.nickname)
        ax2.annotate(
            "5B model > 14B model\n(768x1360 vs 480x832)",
            xy=(cog_idx, t2v[cog_idx].energy_per_generation_joules / 1000),
            xytext=(cog_idx - 1.2, t2v[cog_idx].energy_per_generation_joules / 1000 + 250),
            fontsize=11,
            ha='center',
            arrowprops=dict(arrowstyle='->', color='gray', lw=1),
        )

        fig.tight_layout()
        return fig

    savefig(draw, "section1-3-diffusion")


def _best_per_model(runs: DiffusionRuns) -> list[DiffusionRun]:
    """Minimum-energy configuration per model, sorted by energy."""
    best = [
        min(group, key=lambda r: r.energy_per_generation_joules)
        for group in runs.group_by("nickname").values()
    ]
    return sorted(best, key=lambda r: r.energy_per_generation_joules)


def _analyze_text_to_image() -> list[DiffusionRun]:
    results = _best_per_model(_diff().task("text-to-image").gpu("B200"))
    for r in results:
        if r.inference_steps is None:
            raise ValueError(f"Missing inference_steps for {r.nickname}")
    ratio = results[-1].energy_per_generation_joules / results[0].energy_per_generation_joules
    print(f"Text-to-image range: {results[0].energy_per_generation_joules:.0f} J to "
          f"{results[-1].energy_per_generation_joules:.0f} J = {ratio:.0f}x")
    return results


def _analyze_text_to_video() -> list[DiffusionRun]:
    results = _best_per_model(_diff().task("text-to-video").gpu("B200"))
    print(f"Text-to-video range: {results[0].energy_per_generation_joules/1000:.0f} kJ to "
          f"{results[-1].energy_per_generation_joules/1000:.0f} kJ")
    return results


def section2_1_batch_size() -> None:
    """Batch size vs energy, throughput, latency, and power."""
    print_header("SECTION 2.1: Batch Size")

    gpqa_runs = _llm().task("gpqa")
    r1_configs = sorted(
        gpqa_runs.nickname("DeepSeek R1").gpu("B200").num_gpus(8).batch(min=MIN_BATCH_SIZE),
        key=lambda r: r.max_num_seqs,
    )

    fim_runs = _llm().task("sourcegraph-fim")
    coder_configs = sorted(
        fim_runs.nickname("Qwen 3 Coder 30B A3B").gpu("B200").num_gpus(1).batch(min=MIN_BATCH_SIZE),
        key=lambda r: r.max_num_seqs,
    )

    for label, configs in [("DeepSeek R1 8xB200", r1_configs), ("Qwen 3 Coder 30B A3B 1xB200", coder_configs)]:
        if len(configs) >= 2:
            e_reduction = configs[0].energy_per_token_joules / configs[-1].energy_per_token_joules
            print(f"{label}: {e_reduction:.1f}x energy reduction from batch {configs[0].max_num_seqs} to {configs[-1].max_num_seqs}")

    def draw():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), tight_layout=True)
        ax1.set_axisbelow(True)
        ax2.set_axisbelow(True)

        batches = [r.max_num_seqs for r in r1_configs]
        energies = [r.energy_per_token_joules for r in r1_configs]
        throughputs = [r.output_throughput_tokens_per_sec for r in r1_configs]
        latencies = [r.median_itl_ms for r in r1_configs]
        powers = [r.avg_power_watts for r in r1_configs]

        num_gpus_1 = r1_configs[0].num_gpus
        tdp_1 = num_gpus_1 * 1000  # B200 TDP is 1000W
        e_norm = [e / max(energies) * 100 for e in energies]
        t_norm = [t / max(throughputs) * 100 for t in throughputs]
        l_norm = [l / max(latencies) * 100 for l in latencies]
        p_norm = [p / tdp_1 * 100 for p in powers]

        ax1.grid(True, alpha=0.3)
        ax1.plot(batches, e_norm, "o-", color=COLORS["red"], linewidth=2, markersize=8, label="Energy/Token")
        ax1.plot(batches, t_norm, "s--", color=COLORS["blue"], linewidth=2, markersize=8, label="Tokens/Second")
        ax1.plot(batches, l_norm, "^:", color=COLORS["green"], linewidth=2, markersize=8, label="Median ITL")
        ax1.plot(batches, p_norm, "d-.", color=COLORS["purple"], linewidth=2, markersize=8, label="Power")
        ax1.set_xlabel("Batch Size", fontsize=13)
        ax1.set_ylabel("% of Maximum", fontsize=13)
        ax1.tick_params(axis="y", labelsize=11)
        ax1.tick_params(axis="x", labelsize=11)
        ax1.set_title("DeepSeek R1 (Problem Solving) on 8x B200", fontsize=12)
        ax1.legend(fontsize=10, loc="lower center")
        ax1.set_ylim(0, 110)

        batches2 = [r.max_num_seqs for r in coder_configs]
        energies2 = [r.energy_per_token_joules for r in coder_configs]
        throughputs2 = [r.output_throughput_tokens_per_sec for r in coder_configs]
        latencies2 = [r.median_itl_ms for r in coder_configs]
        powers2 = [r.avg_power_watts for r in coder_configs]

        num_gpus_2 = coder_configs[0].num_gpus
        tdp_2 = num_gpus_2 * 1000  # B200 TDP is 1000W
        e_norm2 = [e / max(energies2) * 100 for e in energies2]
        t_norm2 = [t / max(throughputs2) * 100 for t in throughputs2]
        l_norm2 = [l / max(latencies2) * 100 for l in latencies2]
        p_norm2 = [p / tdp_2 * 100 for p in powers2]

        ax2.grid(True, alpha=0.3)
        ax2.plot(batches2, e_norm2, "o-", color=COLORS["red"], linewidth=2, markersize=8, label="Energy/Token")
        ax2.plot(batches2, t_norm2, "s--", color=COLORS["blue"], linewidth=2, markersize=8, label="Tokens/Second")
        ax2.plot(batches2, l_norm2, "^:", color=COLORS["green"], linewidth=2, markersize=8, label="Median ITL")
        ax2.plot(batches2, p_norm2, "d-.", color=COLORS["purple"], linewidth=2, markersize=8, label="Power")
        ax2.set_xlabel("Batch Size", fontsize=13)
        ax2.set_ylabel("% of Maximum", fontsize=13)
        ax2.tick_params(axis="y", labelsize=11)
        ax2.tick_params(axis="x", labelsize=11)
        ax2.set_title("Qwen 3 Coder 30B A3B (Code Completion) on 1x B200", fontsize=12)
        ax2.legend(fontsize=10, loc="center right")
        ax2.set_ylim(0, 110)

        return fig

    savefig(draw, "section2-1-batch-size")


def section2_2_moe_energy() -> None:
    """MoE vs dense: energy scales with active parameters."""
    print_header("SECTION 2.2: Model Size and Architecture")

    gpqa_runs = _llm().task("gpqa")

    target_configs = [
        ("Qwen 3 8B", "B200", 1),
        ("Qwen 3 14B", "B200", 1),
        ("Qwen 3 32B", "B200", 1),
        ("Qwen 3 30B A3B Thinking", "B200", 1),
        ("Qwen 3 235B A22B Thinking", "B200", 8),
    ]

    selected_runs: list[LLMRun] = []
    for model, gpu, n_gpus in target_configs:
        filtered = gpqa_runs.nickname(model).gpu(gpu).num_gpus(n_gpus).precision("bfloat16")
        if not filtered:
            continue
        selected_runs.append(min(filtered, key=lambda r: r.energy_per_token_joules))

    moe_3b = next((r for r in selected_runs if r.nickname.startswith("Qwen 3 30B A3B Thinking")), None)
    dense_32b = next((r for r in selected_runs if r.nickname.startswith("Qwen 3 32B")), None)
    if moe_3b and dense_32b:
        ratio = dense_32b.energy_per_token_joules / moe_3b.energy_per_token_joules
        print(f"30B A3B MoE: {moe_3b.energy_per_token_joules:.4f} J/tok vs 32B dense: {dense_32b.energy_per_token_joules:.4f} J/tok = {ratio:.2f}x lower")

    def draw():
        fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)
        ax.set_axisbelow(True)

        label_offsets = {
            "Qwen 3 30B A3B": (10, -10),
            "Qwen 3 8B": (-15, 10),
            "Qwen 3 14B": (0, 10),
            "Qwen 3 235B A22B": (5, 10),
            "Qwen 3 32B": (0, 10),
        }

        for run in selected_runs:
            is_moe = run.architecture == "MoE"
            color = COLORS["red"] if is_moe else COLORS["blue"]
            ax.scatter(run.activated_params_billions, run.energy_per_token_joules, s=150, c=color,
                      edgecolors="black", linewidth=1)
            display_label = f"{run.nickname}\n({run.num_gpus}x {run.gpu_model})"
            label = display_label.replace(" Thinking", "").replace(" Instruct", "")
            if run.nickname.startswith("Qwen 3 30B A3B"):
                label = label.replace("\n", " ")
            offset = (8, 5)
            for model_prefix, off in label_offsets.items():
                if run.nickname.startswith(model_prefix):
                    offset = off
                    break
            ax.annotate(label, (run.activated_params_billions, run.energy_per_token_joules),
                        xytext=offset, textcoords="offset points", fontsize=9)

        ax.set_xlabel("Active Parameters (Billions)", fontsize=11)
        ax.set_ylabel("Energy per Token (J)", fontsize=11)
        ax.set_title("Energy per Token by Active Parameters", fontsize=13)
        ax.tick_params(axis='both', labelsize=10)
        ax.set_xlim(0, 40)
        ax.set_ylim(0, 0.85)
        ax.grid(True, alpha=0.3)

        legend_elements = [
            Patch(facecolor=COLORS["red"], label="MoE"),
            Patch(facecolor=COLORS["blue"], label="Dense"),
        ]
        ax.legend(handles=legend_elements, loc="upper left", fontsize=10)

        return fig

    savefig(draw, "section2-2-model-size")


def section2_3_b200_vs_h100() -> None:
    """B200 vs H100 energy at matched latency constraints."""
    print_header("SECTION 2.3: GPU Generation (B200 vs H100)")

    task_display_names = {
        "gpqa": "Problem Solving",
        "lm-arena-chat": "Text Conversation",
        "sourcegraph-fim": "Code Infilling",
    }

    def find_best_llm_for_deadline(
        runs: LLMRuns, model: str, gpu: str, precision: str, deadline: float,
        latency_attr: str, energy_attr: str,
    ) -> LLMRun | None:
        matching = [
            r for r in runs
            if r.nickname == model and r.gpu_model == gpu
            and r.weight_precision == precision
            and getattr(r, latency_attr) <= deadline
        ]
        return min(matching, key=lambda r: getattr(r, energy_attr)) if matching else None

    def find_best_diff_for_deadline(
        runs: DiffusionRuns, model: str, gpu: str, precision: str, deadline: float,
    ) -> DiffusionRun | None:
        matching = [
            r for r in runs
            if r.nickname == model and r.gpu_model == gpu
            and r.weight_precision == precision
            and r.batch_latency_s <= deadline
        ]
        return min(matching, key=lambda r: r.energy_per_generation_joules) if matching else None

    def collect_llm_comparisons(deadlines: list[int]) -> list[IsoTimeComparison]:
        comparisons: list[IsoTimeComparison] = []
        for task_id in ["gpqa", "lm-arena-chat"]:
            task_runs = _llm().task(task_id)
            pairs = {(r.nickname, r.weight_precision) for r in task_runs}
            for model, precision in sorted(pairs):
                has_both = (
                    any(r.gpu_model == "H100" for r in task_runs if r.nickname == model and r.weight_precision == precision)
                    and any(r.gpu_model == "B200" for r in task_runs if r.nickname == model and r.weight_precision == precision)
                )
                if not has_both:
                    continue
                for d in deadlines:
                    h = find_best_llm_for_deadline(task_runs, model, "H100", precision, d, "median_itl_ms", "energy_per_token_joules")
                    b = find_best_llm_for_deadline(task_runs, model, "B200", precision, d, "median_itl_ms", "energy_per_token_joules")
                    if h and b:
                        gain = (h.energy_per_token_joules - b.energy_per_token_joules) / h.energy_per_token_joules * 100
                        comparisons.append(IsoTimeComparison(model, task_id, d, h.energy_per_token_joules, b.energy_per_token_joules, gain))
        return comparisons

    def collect_diff_comparisons(task_id: str, deadlines: list[int]) -> list[IsoTimeComparison]:
        comparisons: list[IsoTimeComparison] = []
        task_runs = _diff().task(task_id)
        pairs = {(r.nickname, r.weight_precision) for r in task_runs}
        for model, precision in sorted(pairs):
            has_both = (
                any(r.gpu_model == "H100" for r in task_runs if r.nickname == model and r.weight_precision == precision)
                and any(r.gpu_model == "B200" for r in task_runs if r.nickname == model and r.weight_precision == precision)
            )
            if not has_both:
                continue
            for d in deadlines:
                h = find_best_diff_for_deadline(task_runs, model, "H100", precision, d)
                b = find_best_diff_for_deadline(task_runs, model, "B200", precision, d)
                if h and b:
                    gain = (h.energy_per_generation_joules - b.energy_per_generation_joules) / h.energy_per_generation_joules * 100
                    comparisons.append(IsoTimeComparison(model, task_id, d, h.energy_per_generation_joules, b.energy_per_generation_joules, gain))
        return comparisons

    llm_deadlines = [50, 100, 250]
    image_deadlines = [10, 30, 60]
    video_deadlines = [100, 500, 1000]

    llm_comparisons = collect_llm_comparisons(llm_deadlines)
    image_comparisons = collect_diff_comparisons("text-to-image", image_deadlines)
    video_comparisons = collect_diff_comparisons("text-to-video", video_deadlines)

    for label, comps in [("LLM", llm_comparisons), ("Text-to-Image", image_comparisons), ("Text-to-Video", video_comparisons)]:
        if comps:
            b200_wins = sum(1 for c in comps if c.gain_pct > 0)
            gains = [c.gain_pct for c in comps]
            print(f"{label}: B200 wins {b200_wins}/{len(comps)}, median {np.median(gains):.0f}% "
                  f"(range {min(gains):.0f}% to {max(gains):.0f}%)")

    def get_spread_gains(comparisons: list[IsoTimeComparison], n: int = 4, min_gain: float = 0) -> list[IsoTimeComparison]:
        """Select n comparisons with equally spaced positive gains for plotting."""
        b200_wins = [c for c in comparisons if c.gain_pct >= min_gain]
        if len(b200_wins) <= n:
            return sorted(b200_wins, key=lambda x: x.gain_pct, reverse=True)
        sorted_by_gain = sorted(b200_wins, key=lambda x: x.gain_pct, reverse=True)
        indices = [int(i * (len(sorted_by_gain) - 1) / (n - 1)) for i in range(n)]
        return [sorted_by_gain[i] for i in indices]

    llm_plot_data = get_spread_gains(llm_comparisons, 4, min_gain=10)
    image_plot_data = get_spread_gains(image_comparisons, 4)
    video_plot_data = get_spread_gains(video_comparisons, 4)

    def draw():
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 5))
        ax1.set_axisbelow(True)
        ax2.set_axisbelow(True)
        ax3.set_axisbelow(True)

        width = 0.35

        models1 = [f"{c.model.replace(' Thinking', '')}\n{task_display_names.get(c.task, c.task)}" for c in llm_plot_data]
        h100_vals1 = [c.h100_energy for c in llm_plot_data]
        b200_vals1 = [c.b200_energy for c in llm_plot_data]

        x1 = np.arange(len(models1))

        ax1.grid(True, axis='y', alpha=0.3)
        ax1.bar(x1 - width/2, h100_vals1, width, label="H100", color=COLORS["red"])
        ax1.bar(x1 + width/2, b200_vals1, width, label="B200", color=COLORS["blue"])
        ax1.set_ylabel("Energy per Token (J)", fontsize=11)
        ax1.set_title(f"LLMs (Median ITL $\\leq$ {llm_deadlines[1]} ms)", fontsize=15)
        ax1.set_xticks(x1)
        ax1.set_xticklabels(models1, rotation=30, ha="center", fontsize=10)
        ax1.tick_params(axis='y', labelsize=11)
        ax1.legend(fontsize=10)
        ax1.set_ylim(0, max(h100_vals1) * 1.15)

        for i, c in enumerate(llm_plot_data):
            is_zero = round(c.gain_pct) == 0
            label = "0%" if is_zero else f"-{c.gain_pct:.0f}%"
            ax1.annotate(label,
                        xy=(i + width/2, c.b200_energy),
                        xytext=(0 if is_zero else 5, 5), textcoords="offset points",
                        ha="center", fontsize=9)

        if image_plot_data:
            models2 = [c.model for c in image_plot_data]
            h100_vals2 = [c.h100_energy for c in image_plot_data]
            b200_vals2 = [c.b200_energy for c in image_plot_data]

            x2 = np.arange(len(models2))

            ax2.grid(True, axis='y', alpha=0.3)
            ax2.bar(x2 - width/2, h100_vals2, width, label="H100", color=COLORS["red"])
            ax2.bar(x2 + width/2, b200_vals2, width, label="B200", color=COLORS["blue"])
            ax2.set_ylabel("Energy per Image (J)", fontsize=11)
            ax2.set_title(f"Text-to-Image (Gen. Time $\\leq$ {image_deadlines[1]} s)", fontsize=15)
            ax2.set_xticks(x2)
            ax2.set_xticklabels(models2, rotation=30, ha="center", fontsize=10)
            ax2.tick_params(axis='y', labelsize=11)
            ax2.legend(fontsize=10)
            ax2.set_ylim(0, max(h100_vals2) * 1.15)

            for i, c in enumerate(image_plot_data):
                is_zero = round(c.gain_pct) == 0
                label = "0%" if is_zero else f"-{c.gain_pct:.0f}%"
                ax2.annotate(label,
                            xy=(i + width/2, c.b200_energy),
                            xytext=(0 if is_zero else 5, 5), textcoords="offset points",
                            ha="center", fontsize=9)

        if video_plot_data:
            models3 = [c.model for c in video_plot_data]
            h100_vals3 = [c.h100_energy for c in video_plot_data]
            b200_vals3 = [c.b200_energy for c in video_plot_data]

            x3 = np.arange(len(models3))

            ax3.grid(True, axis='y', alpha=0.3)
            ax3.bar(x3 - width/2, h100_vals3, width, label="H100", color=COLORS["red"])
            ax3.bar(x3 + width/2, b200_vals3, width, label="B200", color=COLORS["blue"])
            ax3.set_ylabel("Energy per Video (J)", fontsize=11)
            ax3.set_title(f"Text-to-Video (Gen. Time $\\leq$ {video_deadlines[1]} s)", fontsize=15)
            ax3.set_xticks(x3)
            ax3.set_xticklabels(models3, rotation=30, ha="center", fontsize=10)
            ax3.tick_params(axis='y', labelsize=11)
            ax3.legend(fontsize=10)
            ax3.set_ylim(0, max(h100_vals3) * 1.15)

            for i, c in enumerate(video_plot_data):
                is_zero = round(c.gain_pct) == 0
                label = "0%" if is_zero else f"-{c.gain_pct:.0f}%"
                ax3.annotate(label,
                            xy=(i + width/2, c.b200_energy),
                            xytext=(0 if is_zero else 5, 5), textcoords="offset points",
                            ha="center", fontsize=9)

        fig.tight_layout()
        return fig

    savefig(draw, "section2-3-b200-vs-h100")


def section2_4_precision() -> None:
    """FP8 vs BF16: FP8 wins at larger batch sizes."""
    print_header("SECTION 2.4: Precision")

    comparisons = _analyze_fp8_vs_bf16()

    typical_case = [c for c in comparisons
                    if "235B A22B Instruct" in c.model and "Thinking" not in c.model
                    and c.gpu == "H100" and c.num_gpus == 8 and c.task == "lm-arena-chat"]
    typical_sorted = sorted(typical_case, key=lambda x: x.batch)
    assert len(typical_sorted) == 7, f"Expected 7 FP8/BF16 comparison pairs, got {len(typical_sorted)}"

    def draw():
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), tight_layout=True)

        if typical_sorted:
            batches_t = [c.batch for c in typical_sorted]
            fp8_e = [c.fp8_energy for c in typical_sorted]
            bf16_e = [c.bf16_energy for c in typical_sorted]
            fp8_l = [c.fp8_latency for c in typical_sorted]
            bf16_l = [c.bf16_latency for c in typical_sorted]

            ax1 = axes[0]
            ax1.grid(True, alpha=0.3)
            ax1.plot(batches_t, fp8_e, "o-", color=COLORS["red"], linewidth=2, markersize=8, label="FP8")
            ax1.plot(batches_t, bf16_e, "s-", color=COLORS["blue"], linewidth=2, markersize=8, label="BF16")
            ax1.axvline(x=32, color="gray", linestyle="--", alpha=0.5)
            ax1.set_xlabel("Batch Size", fontsize=13)
            ax1.set_ylabel("Energy per Token (J)", fontsize=13)
            ax1.set_title("Qwen 3 235B A22B (Text Conversation) on 8x H100", fontsize=12)
            ax1.tick_params(axis='both', labelsize=11)
            ax1.set_ylim(bottom=0)
            ax1.legend(fontsize=10)

            ax2 = axes[1]
            ax2.grid(True, alpha=0.3)
            ax2.plot(batches_t, fp8_l, "o-", color=COLORS["red"], linewidth=2, markersize=8, label="FP8")
            ax2.plot(batches_t, bf16_l, "s-", color=COLORS["blue"], linewidth=2, markersize=8, label="BF16")
            ax2.axvline(x=32, color="gray", linestyle="--", alpha=0.5)
            ax2.set_xlabel("Batch Size", fontsize=13)
            ax2.set_ylabel("Median ITL (ms)", fontsize=13)
            ax2.set_title("Qwen 3 235B A22B (Text Conversation) on 8x H100", fontsize=12)
            ax2.tick_params(axis='both', labelsize=11)
            ax2.set_ylim(bottom=0)
            ax2.legend(fontsize=10)

        return fig

    savefig(draw, "section2-4-precision")


def _analyze_fp8_vs_bf16() -> list[FP8BF16Comparison]:
    """Match FP8 and BF16 runs, print batch-size-binned win rates (blog tables)."""
    comparisons: list[FP8BF16Comparison] = []

    # LLM comparisons
    for task_id, task_runs in _llm().group_by("task").items():
        groups: dict[tuple, dict[str, LLMRun]] = {}
        for r in task_runs:
            base_name = r.nickname.replace(" FP8", "").replace(" fp8", "")
            key = (base_name, r.gpu_model, r.num_gpus, r.max_num_seqs)
            groups.setdefault(key, {})[r.weight_precision] = r

        for key, prec_runs in groups.items():
            if "fp8" in prec_runs and "bfloat16" in prec_runs:
                fp8, bf16 = prec_runs["fp8"], prec_runs["bfloat16"]
                if key[3] < MIN_BATCH_SIZE:
                    continue
                comparisons.append(FP8BF16Comparison(
                    task=task_id, model=key[0], gpu=key[1], num_gpus=key[2], batch=key[3],
                    fp8_energy=fp8.energy_per_token_joules, bf16_energy=bf16.energy_per_token_joules,
                    energy_ratio=fp8.energy_per_token_joules / bf16.energy_per_token_joules,
                    fp8_latency=fp8.median_itl_ms, bf16_latency=bf16.median_itl_ms,
                    latency_ratio=fp8.median_itl_ms / bf16.median_itl_ms if bf16.median_itl_ms > 0 else 1.0,
                ))

    # Diffusion comparisons
    for task_id, task_runs in _diff().group_by("task").items():
        groups_d: dict[tuple, dict[str, DiffusionRun]] = {}
        for r in task_runs:
            base_name = r.nickname.replace(" FP8", "").replace(" fp8", "")
            key = (base_name, r.gpu_model, r.num_gpus, r.batch_size)
            groups_d.setdefault(key, {})[r.weight_precision] = r

        for key, prec_runs in groups_d.items():
            if "fp8" in prec_runs and "bfloat16" in prec_runs:
                fp8, bf16 = prec_runs["fp8"], prec_runs["bfloat16"]
                comparisons.append(FP8BF16Comparison(
                    task=task_id, model=key[0], gpu=key[1], num_gpus=key[2], batch=key[3],
                    fp8_energy=fp8.energy_per_generation_joules, bf16_energy=bf16.energy_per_generation_joules,
                    energy_ratio=fp8.energy_per_generation_joules / bf16.energy_per_generation_joules,
                    fp8_latency=fp8.batch_latency_s, bf16_latency=bf16.batch_latency_s,
                    latency_ratio=fp8.batch_latency_s / bf16.batch_latency_s if bf16.batch_latency_s > 0 else 1.0,
                ))

    # Batch-size-binned summary (directly cited in blog tables)
    filtered = [c for c in comparisons if "480B" not in c.model]
    for low, high, label in [(8, 16, "8-16"), (17, 64, "17-64"), (65, 256, "65-256")]:
        in_range = [c for c in filtered if low <= c.batch <= high]
        if in_range:
            e_wins = sum(1 for c in in_range if c.energy_ratio < 1)
            l_wins = sum(1 for c in in_range if c.latency_ratio < 1)
            total = len(in_range)
            e_diffs = sorted([(c.energy_ratio - 1) * 100 for c in in_range])
            l_diffs = sorted([(c.latency_ratio - 1) * 100 for c in in_range])
            print(f"Batch {label}: energy FP8 wins {e_wins}/{total} (median {e_diffs[len(e_diffs)//2]:+.0f}%, "
                  f"range {min(e_diffs):+.0f}% to {max(e_diffs):+.0f}%), "
                  f"latency {l_wins}/{total} (median {l_diffs[len(l_diffs)//2]:+.0f}%)")

    return comparisons


def section2_5_multi_gpu_scaling() -> None:
    """Multi-GPU scaling: time-energy tradeoffs and capacity unlock."""
    print_header("SECTION 2.5: Multi-GPU Scaling")

    gpqa_runs = _llm().task("gpqa")
    model_runs = gpqa_runs.nickname("GPT OSS 120B").gpu("B200")

    gpu1_runs = sorted(
        model_runs.num_gpus(1).batch(min=MIN_BATCH_SIZE),
        key=lambda r: r.max_num_seqs,
    )
    gpu2_runs = sorted(
        model_runs.num_gpus(2).batch(min=MIN_BATCH_SIZE),
        key=lambda r: r.max_num_seqs,
    )
    assert len(gpu1_runs) == 9, f"Expected 9 GPT OSS 120B B200 1-GPU configs, got {len(gpu1_runs)}"
    assert len(gpu2_runs) == 10, f"Expected 10 GPT OSS 120B B200 2-GPU configs, got {len(gpu2_runs)}"

    # Same-batch, more-GPUs analysis (blog: "81% of cases" / "93% of cases")
    all_runs = _llm().task("gpqa", "lm-arena-chat", "sourcegraph-fim")

    model_groups: dict[tuple[str, str], dict[int, list[LLMRun]]] = defaultdict(lambda: defaultdict(list))
    for r in all_runs:
        model_groups[(r.nickname, r.gpu_model)][r.num_gpus].append(r)

    for gpu_type in ["B200", "H100"]:
        energy_up = 0
        latency_down = 0
        total = 0
        for (nickname, gpu), gpu_runs_by_count in model_groups.items():
            if gpu != gpu_type:
                continue
            gpu_counts = sorted(gpu_runs_by_count.keys())
            for i, n_gpus in enumerate(gpu_counts):
                for m_gpus in gpu_counts[i+1:]:
                    batches_n = {r.max_num_seqs: r for r in gpu_runs_by_count[n_gpus]}
                    batches_m = {r.max_num_seqs: r for r in gpu_runs_by_count[m_gpus]}
                    for batch in set(batches_n) & set(batches_m):
                        total += 1
                        if batches_m[batch].energy_per_token_joules >= batches_n[batch].energy_per_token_joules:
                            energy_up += 1
                        if batches_m[batch].median_itl_ms < batches_n[batch].median_itl_ms:
                            latency_down += 1
        if total > 0:
            print(f"{gpu_type}: same batch + more GPUs -> energy increases {energy_up}/{total} ({100*energy_up/total:.0f}%), "
                  f"latency decreases {latency_down}/{total} ({100*latency_down/total:.0f}%)")

    h100_model_runs = all_runs.nickname("GPT OSS 120B").gpu("H100")
    h100_gpu1_runs = sorted(
        h100_model_runs.num_gpus(1).batch(min=MIN_BATCH_SIZE),
        key=lambda r: r.max_num_seqs,
    )
    h100_gpu2_runs = sorted(
        h100_model_runs.num_gpus(2).batch(min=MIN_BATCH_SIZE),
        key=lambda r: r.max_num_seqs,
    )
    assert len(h100_gpu1_runs) == 3, f"Expected 3 GPT OSS 120B H100 1-GPU configs, got {len(h100_gpu1_runs)}"
    assert len(h100_gpu2_runs) == 13, f"Expected 13 GPT OSS 120B H100 2-GPU configs, got {len(h100_gpu2_runs)}"

    annot_config = {
        ("B200", 1): {
            16: (5, 2),
            64: (5, 2),
            512: (-15, -13),
            2048: (-15, -13),
            3072: (-15, -13),
        },
        ("B200", 2): {
            16: (5, 5),
            64: (5, 5),
            512: (5, 5),
            2048: (5, 5),
            3072: (5, 5),
            4096: (5, 5),
        },
        ("H100", 1): {
            16: (5, 5),
            32: (5, 5),
            64: (5, 5),
        },
        ("H100", 2): {
            16: (-18, -5),
            32: (-19, -5),
            64: (-20, -5),
            256: (-15, -15),
            512: (5, 5),
            1024: (5, 5),
            2048: (5, 8),
        },
    }

    def draw():
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), tight_layout=True)

        ax1 = axes[0]
        ax1.set_axisbelow(True)
        ax1.grid(True, alpha=0.3)

        lat1 = [r.median_itl_ms for r in gpu1_runs]
        eng1 = [r.energy_per_token_joules for r in gpu1_runs]
        batch1 = [r.max_num_seqs for r in gpu1_runs]
        ax1.plot(lat1, eng1, "o-", color=COLORS["blue"], linewidth=2, markersize=8, label="1 GPU", zorder=3)

        lat2 = [r.median_itl_ms for r in gpu2_runs]
        eng2 = [r.energy_per_token_joules for r in gpu2_runs]
        batch2 = [r.max_num_seqs for r in gpu2_runs]
        ax1.plot(lat2, eng2, "s-", color=COLORS["red"], linewidth=2, markersize=8, label="2 GPUs", zorder=3)

        for i, (x, y, b) in enumerate(zip(lat1, eng1, batch1)):
            if b in annot_config.get(("B200", 1), {}):
                ox, oy = annot_config[("B200", 1)][b]
                ax1.annotate(str(b), (x, y), textcoords="offset points", xytext=(ox, oy), fontsize=9, color=COLORS["blue"])
        for i, (x, y, b) in enumerate(zip(lat2, eng2, batch2)):
            if b in annot_config.get(("B200", 2), {}):
                ox, oy = annot_config[("B200", 2)][b]
                ax1.annotate(str(b), (x, y), textcoords="offset points", xytext=(ox, oy), fontsize=9, color=COLORS["red"])

        ax1.set_xlabel("Median ITL (ms)", fontsize=13)
        ax1.set_ylabel("Energy per Token (J)", fontsize=13)
        ax1.set_title("GPT OSS 120B (Problem Solving) on B200", fontsize=12)
        ax1.tick_params(axis="both", labelsize=11)
        ax1.legend(fontsize=10, loc="upper right")
        ax1.set_xlim(left=0, right=235)
        ax1.set_ylim(bottom=0, top=0.5)

        ax2 = axes[1]
        ax2.set_axisbelow(True)
        ax2.grid(True, alpha=0.3)

        if h100_gpu1_runs and h100_gpu2_runs:
            lat1_h = [r.median_itl_ms for r in h100_gpu1_runs]
            eng1_h = [r.energy_per_token_joules for r in h100_gpu1_runs]
            batch1_h = [r.max_num_seqs for r in h100_gpu1_runs]
            ax2.plot(lat1_h, eng1_h, "o-", color=COLORS["blue"], linewidth=2, markersize=8, label="1 GPU", zorder=3)

            lat2_h = [r.median_itl_ms for r in h100_gpu2_runs]
            eng2_h = [r.energy_per_token_joules for r in h100_gpu2_runs]
            batch2_h = [r.max_num_seqs for r in h100_gpu2_runs]
            ax2.plot(lat2_h, eng2_h, "s-", color=COLORS["red"], linewidth=2, markersize=8, label="2 GPUs", zorder=3)

            for i, (x, y, b) in enumerate(zip(lat1_h, eng1_h, batch1_h)):
                if b in annot_config.get(("H100", 1), {}):
                    ox, oy = annot_config[("H100", 1)][b]
                    ax2.annotate(str(b), (x, y), textcoords="offset points", xytext=(ox, oy), fontsize=9, color=COLORS["blue"])
            for i, (x, y, b) in enumerate(zip(lat2_h, eng2_h, batch2_h)):
                if b in annot_config.get(("H100", 2), {}):
                    ox, oy = annot_config[("H100", 2)][b]
                    ax2.annotate(str(b), (x, y), textcoords="offset points", xytext=(ox, oy), fontsize=9, color=COLORS["red"])

        ax2.set_xlabel("Median ITL (ms)", fontsize=13)
        ax2.set_ylabel("Energy per Token (J)", fontsize=13)
        ax2.set_title("GPT OSS 120B (Problem Solving) on H100", fontsize=12)
        ax2.tick_params(axis="both", labelsize=11)
        ax2.legend(fontsize=10, loc="upper right")
        ax2.set_xlim(left=0, right=140)
        ax2.set_ylim(bottom=0, top=0.65)

        return fig

    savefig(draw, "section2-5-multi-gpu")

    # 235B case study time-energy tradeoff
    case_study_model = "Qwen 3 235B A22B Thinking FP8"
    case_study_task = "gpqa"
    task_display_names = {"gpqa": "Problem Solving"}

    case_runs = list(_llm().task(case_study_task).nickname(case_study_model).batch(min=MIN_BATCH_SIZE))
    case_gpu_runs: dict[tuple[str, int], list[LLMRun]] = {}
    for r in case_runs:
        case_gpu_runs.setdefault((r.gpu_model, r.num_gpus), []).append(r)
    print(f"235B case study configs: {sorted(case_gpu_runs.keys())}")

    def draw_235b_tradeoff():
        fig, ax = plt.subplots(figsize=(5, 4), tight_layout=True)
        ax.set_axisbelow(True)
        ax.grid(True, alpha=0.3)

        markers = ['o', 's', '^', 'D']
        colors_list = [COLORS["red"], COLORS["blue"], COLORS["green"], COLORS["purple"]]

        batch_annot_config = {
            ("B200", 2): {
                0: (7, -3),
                1: (7, -3),
                2: (7, -3),
            },
            ("B200", 4): {
                0: (-20, -3),
                1: (-20, -3),
                2: (-20, -3),
                3: (-22, -14),
                4: (-15, -15),
                5: (-10, 8),
                6: (-5, 8),
                7: (-15, 8),
            },
            ("H100", 4): {
                0: (-8, 9),
                1: (-16, 1),
            },
            ("H100", 8): {
                0: (-20, -3),
                1: (7, -4),
                2: (-18, -3),
                4: (-19, -12),
            },
        }

        for idx, ((gpu, num_gpus), runs) in enumerate(sorted(case_gpu_runs.items())):
            sorted_runs = sorted(runs, key=lambda r: r.max_num_seqs)
            batches = [r.max_num_seqs for r in sorted_runs]
            itls = [r.median_itl_ms for r in sorted_runs]
            energies = [r.energy_per_token_joules for r in sorted_runs]

            label = f"{num_gpus}x {gpu}"
            color = colors_list[idx % len(colors_list)]
            marker = markers[idx % len(markers)]
            ax.plot(itls, energies, marker=marker, label=label, color=color, markersize=7, linewidth=2)

            annot_indices = batch_annot_config.get((gpu, num_gpus), {})
            for i, (ox, oy) in annot_indices.items():
                if i < len(batches):
                    ax.annotate(f"{batches[i]}", (itls[i], energies[i]),
                               xytext=(ox, oy), textcoords="offset points",
                               fontsize=9, color=color)

        ax.set_xlabel('Median ITL (ms)', fontsize=11)
        ax.set_ylabel('Energy per Token (J)', fontsize=11)
        ax.set_title(f'{case_study_model} ({task_display_names[case_study_task]})', fontsize=12)
        ax.legend(fontsize=10, loc='upper right', ncol=2)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0, top=7)
        ax.tick_params(axis='both', labelsize=10)

        return fig

    savefig(draw_235b_tradeoff, "section2-5-235b-tradeoff")


def section3_power() -> None:
    """Energy per token and throughput per watt vs batch size."""
    print_header("SECTION 3: Power")

    chat_runs = _llm().task("lm-arena-chat")

    models = [
        ("Qwen 3 8B", 1, COLORS["blue"]),
        ("Llama 3.1 70B Instruct", 2, COLORS["green"]),
        ("Qwen 3 235B A22B Instruct", 4, COLORS["red"]),
        ("Llama 3.1 405B Instruct", 8, COLORS["purple"]),
    ]

    def draw():
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), tight_layout=True)

        for idx, (model_name, num_gpus, color) in enumerate(models):
            model_runs = sorted(
                chat_runs.nickname(model_name).gpu("B200").num_gpus(num_gpus).batch(min=MIN_BATCH_SIZE),
                key=lambda r: r.max_num_seqs,
            )

            if not model_runs:
                continue

            batches = [r.max_num_seqs for r in model_runs]
            energy_per_tok = [r.energy_per_token_joules for r in model_runs]
            throughput_per_watt = [r.output_throughput_tokens_per_sec / r.avg_power_watts for r in model_runs]

            label = f"{model_name} ({num_gpus}x B200)"

            ax1 = axes[0]
            ax1.plot(batches, energy_per_tok, "o-", color=color, linewidth=2, markersize=8, label=label)

            ax2 = axes[1]
            ax2.plot(batches, throughput_per_watt, "o-", color=color, linewidth=2, markersize=8, label=label)


        ax1 = axes[0]
        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel("Batch Size", fontsize=13)
        ax1.set_ylabel("Energy per Token (J)", fontsize=13)
        ax1.set_title("Energy per Token", fontsize=12)
        ax1.tick_params(axis='both', labelsize=11)
        ax1.set_yscale("log")
        ax1.legend(fontsize=10, loc="upper right")

        ax2 = axes[1]
        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel("Batch Size", fontsize=13)
        ax2.set_ylabel("Tokens/s/W", fontsize=13)
        ax2.set_title("Token Throughput per Watt", fontsize=12)
        ax2.tick_params(axis='both', labelsize=11)
        ax2.set_ylim(bottom=0)

        return fig

    savefig(draw, "section3-power")


SECTIONS = {
    "1.1": section1_1_llm_tokens,
    "1.2": section1_2_mllm,
    "1.3": section1_3_diffusion,
    "2.1": section2_1_batch_size,
    "2.2": section2_2_moe_energy,
    "2.3": section2_3_b200_vs_h100,
    "2.4": section2_4_precision,
    "2.5": section2_5_multi_gpu_scaling,
    "3": section3_power,
}


def run_all(sections: list[str] | None = None) -> None:
    """Run selected (or all) sections, generating figures."""
    if sections is None:
        sections = list(SECTIONS.keys())

    print("\n" + "=" * 80)
    if len(sections) == len(SECTIONS):
        print("ML.ENERGY LEADERBOARD V3.0 - COMPLETE ANALYSIS")
    else:
        print(f"ML.ENERGY LEADERBOARD V3.0 - SECTIONS: {', '.join(sections)}")
    print("=" * 80 + "\n")

    for section_id in sections:
        if section_id not in SECTIONS:
            print(f"Warning: Unknown section '{section_id}', skipping")
            continue
        SECTIONS[section_id]()

    print("\n" + "=" * 80)
    print(f"ANALYSES COMPLETE - Figures saved as SVG/PNG in {FIGURES_DIR}/")
    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate ML.ENERGY Leaderboard v3.0 blog figures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available sections: {', '.join(SECTIONS.keys())}",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--mlenergy-data-dir",
        help="Compiled data directory (contains runs/llm.parquet).",
    )
    source.add_argument(
        "--results-dir", action="append", dest="results_dirs",
        help="Raw benchmark results directory (repeatable). Mutually exclusive with --mlenergy-data-dir.",
    )
    parser.add_argument("--llm-config-dir", help="LLM config directory (for --results-dir mode).")
    parser.add_argument("--diffusion-config-dir", help="Diffusion config directory (for --results-dir mode).")
    parser.add_argument(
        "sections", nargs="*",
        help="Section IDs to run (e.g., 1.1 2.3). If omitted, runs all.",
    )

    args = parser.parse_args()

    setup(
        mlenergy_data_dir=args.mlenergy_data_dir,
        results_dirs=args.results_dirs,
        llm_config_dir=args.llm_config_dir,
        diffusion_config_dir=args.diffusion_config_dir,
    )
    run_all(args.sections if args.sections else None)
