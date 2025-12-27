"""
ML.ENERGY Leaderboard v3.0 Blog Analysis Scripts

This module provides analysis functions for generating insights from the
ML.ENERGY benchmark data. Functions are organized into three sections:

1. Energy per Response - what determines the cost of one answer
2. Energy per Token - what affects the unit cost of computation
3. Power - how power varies and affects cluster capacity

Usage:
    $ uv run python blog_analysis_scripts.py  # Run all analyses

    Or interactively:
    >>> from blog_analysis_scripts import *
    >>> section1_1_llm_tokens()  # Run analysis for Section 1.1

Data Requirements:
    - JSON data files must be present in public/data/tasks/
    - Run from the leaderboard/ directory
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TextToImageResult:
    """Result for text-to-image analysis."""
    model: str
    params: float
    steps: int
    energy: float
    throughput: float


@dataclass
class TextToVideoResult:
    """Result for text-to-video analysis."""
    model: str
    params: float
    resolution: str
    frames: int
    energy: float
    energy_wh: float


@dataclass
class MLLMResult:
    """Result for multimodal LLM analysis."""
    model: str
    modality: str
    energy_per_response: float
    avg_output_len: float
    energy_per_token: float
    batch_size: int


@dataclass
class Section1_2Result:
    """Result for Section 1.2 MLLM analysis."""
    text_results: list[MLLMResult]
    image_results: list[MLLMResult]
    video_results: list[MLLMResult]


@dataclass
class Section1_3Result:
    """Result for Section 1.3 diffusion analysis."""
    text_to_image: list[TextToImageResult]
    text_to_video: list[TextToVideoResult]


@dataclass
class MoEEnergyResult:
    """Result for MoE vs Dense energy analysis."""
    model: str
    total_params: float
    active_params: float
    energy_per_token: float
    architecture: str
    is_moe: bool


@dataclass
class GPUComparisonResult:
    """Result for GPU comparison (B200 vs H100)."""
    task: str
    model: str
    num_gpus: int
    batch: int
    precision: str
    h100_energy: float
    b200_energy: float
    gain_pct: float


@dataclass
class BatchSizeEnergyResult:
    """Result for batch size energy analysis."""
    batch: int
    energy_per_token: float
    latency_ms: float
    throughput: float


@dataclass
class FP8BF16Comparison:
    """Result for FP8 vs BF16 comparison."""
    task: str
    model: str
    gpu: str
    num_gpus: int
    batch: int
    fp8_energy: float
    bf16_energy: float
    energy_ratio: float  # <1 means FP8 wins
    fp8_wins_energy: bool
    fp8_latency: float
    bf16_latency: float
    latency_ratio: float  # <1 means FP8 wins (faster)
    fp8_wins_latency: bool


@dataclass
class MultiGPUScalingResult:
    """Result for multi-GPU scaling analysis."""
    task: str
    model: str
    precision: str
    gpu: str
    scaling: dict[int, Any]  # num_gpus -> config


@dataclass
class BatchMaxResult:
    """Max batch size by modality."""
    family: str
    text_max: int
    img_max: int
    vid_max: int


@dataclass
class ThroughputPerWattResult:
    """Throughput per watt by modality."""
    family: str
    text_tpw: float | None
    img_tpw: float | None
    vid_tpw: float | None


@dataclass
class Section2_6Result:
    """Result for Section 2.6 multimodal analysis."""
    batch_comparison: list[BatchMaxResult]
    tpw_comparison: list[ThroughputPerWattResult]


@dataclass
class ModelPowerResult:
    """Result for model power analysis."""
    model: str
    active_params: float
    gpu_power: float
    throughput: float
    batch: int
    is_moe: bool


@dataclass
class BatchPowerResult:
    """Result for batch size power analysis."""
    batch: int
    throughput: float
    gpu_power: float
    tpw: float


@dataclass
class ThroughputResult:
    """Result for throughput per watt analysis."""
    model: str
    gpu: str
    num_gpus: int
    batch: int
    throughput_tokens: float
    throughput_requests: float
    avg_output_len: float
    gpu_power: float
    cluster_power: float
    tpw_tokens: float
    tpw_requests: float


@dataclass
class H100B200Comparison:
    """Result for H100 vs B200 throughput comparison."""
    model: str
    batch: int
    h100_tpw_tokens: float
    b200_tpw_tokens: float
    h100_tpw_requests: float
    b200_tpw_requests: float
    gain: float


@dataclass
class Section3_3Result:
    """Result for Section 3.3 throughput per watt analysis."""
    top_models: list[ThroughputResult]
    h100_b200_comparison: list[H100B200Comparison]
    model_classes: dict[str, list[ThroughputResult]]

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = Path("public/data")
TASKS_DIR = DATA_DIR / "tasks"

# Cluster power overhead: GPU power × this factor = total cluster power
# Accounts for CPU, DRAM, networking, cooling, and other infrastructure
CLUSTER_OVERHEAD = 2.0

# ML.ENERGY brand colors
MLENERGY_GREEN = "#23d175"
MLENERGY_BG = "#2e303e"

# Color palette for plots
COLORS = {
    "green": MLENERGY_GREEN,
    "red": "#e74c3c",
    "blue": "#3498db",
    "orange": "#f39c12",
    "purple": "#9b59b6",
    "gray": "gray",
}

# Output directory for figures
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)


# =============================================================================
# Data Loading
# =============================================================================

def load_tasks() -> dict[str, Any]:
    """Load all task data from JSON files."""
    tasks = {}
    for filepath in TASKS_DIR.glob("*.json"):
        with open(filepath) as f:
            tasks[filepath.stem] = json.load(f)
    return tasks


def load_index() -> dict[str, Any]:
    """Load the global index file."""
    with open(DATA_DIR / "index.json") as f:
        return json.load(f)


# Load data at module import time
TASKS = load_tasks()
INDEX = load_index()

print(f"Loaded {len(TASKS)} tasks: {list(TASKS.keys())}")


# =============================================================================
# Helper Functions
# =============================================================================

def get_configs(task_id: str) -> list[dict]:
    """Get configurations for a specific task."""
    return TASKS.get(task_id, {}).get("configurations", [])


def filter_configs(
    configs: list[dict],
    *,
    nickname: str | None = None,
    gpu_model: str | None = None,
    num_gpus: int | None = None,
    precision: str | None = None,
) -> list[dict]:
    """Filter configurations by specified criteria. Required fields must exist."""
    result = configs
    if nickname is not None:
        for c in result:
            if "nickname" not in c:
                raise KeyError(f"Required field 'nickname' missing from config: {c.get('model_id', 'unknown')}")
        result = [c for c in result if c["nickname"] == nickname]
    if gpu_model is not None:
        for c in result:
            if "gpu_model" not in c:
                raise KeyError(f"Required field 'gpu_model' missing from config: {c.get('nickname', 'unknown')}")
        result = [c for c in result if c["gpu_model"] == gpu_model]
    if num_gpus is not None:
        for c in result:
            if "num_gpus" not in c:
                raise KeyError(f"Required field 'num_gpus' missing from config: {c.get('nickname', 'unknown')}")
        result = [c for c in result if c["num_gpus"] == num_gpus]
    if precision is not None:
        for c in result:
            if "weight_precision" not in c:
                raise KeyError(f"Required field 'weight_precision' missing from config: {c.get('nickname', 'unknown')}")
        result = [c for c in result if c["weight_precision"] == precision]
    return result


def require(config: dict, key: str) -> Any:
    """Get a required field from config, raising an error if missing."""
    if key not in config:
        raise KeyError(f"Required field '{key}' missing from config: {config}")
    return config[key]


def format_model_with_params(model: str, params: float) -> str:
    """Add params in parentheses to model name if not already present.

    If the model name already contains a number followed by 'B' (e.g., "8B", "1.5B"),
    return the name as-is. Otherwise, append the params in parentheses.
    """
    if re.search(r'\d+\.?\d*B\b', model):
        return model
    return f"{model} ({params:.1f}B)"


def get_best_config(
    configs: list[dict],
    key: str = "energy_per_token_joules",
    minimize: bool = True,
) -> dict | None:
    """Get the best configuration by a given metric."""
    if not configs:
        return None
    # Verify all configs have the key
    for c in configs:
        if key not in c:
            raise KeyError(f"Required field '{key}' missing from config: {c.get('nickname', c.get('model_id', 'unknown'))}")
    return min(configs, key=lambda x: x[key]) if minimize else max(configs, key=lambda x: x[key])


def get_batch_size(config: dict) -> int:
    """Get batch size from config (handles both max_num_seqs and batch_size)."""
    if "max_num_seqs" in config and config["max_num_seqs"] is not None:
        return config["max_num_seqs"]
    if "batch_size" in config and config["batch_size"] is not None:
        return config["batch_size"]
    raise KeyError(f"Neither 'max_num_seqs' nor 'batch_size' found in config: {config.get('nickname', config.get('model_id', 'unknown'))}")


# Minimum batch size to include in analysis (cut off 8 due to power anomalies)
MIN_BATCH_SIZE = 16




def compute_throughput_per_watt(config: dict, cluster_overhead: float = CLUSTER_OVERHEAD) -> float:
    """Compute throughput per cluster watt."""
    throughput = require(config, "output_throughput_tokens_per_sec")
    gpu_power = require(config, "avg_power_watts")
    if throughput <= 0:
        raise ValueError(f"Invalid throughput {throughput} in config: {config.get('nickname', 'unknown')}")
    if gpu_power <= 0:
        raise ValueError(f"Invalid gpu_power {gpu_power} in config: {config.get('nickname', 'unknown')}")
    return throughput / (gpu_power * cluster_overhead)


def print_header(title: str, width: int = 80) -> None:
    """Print a section header."""
    print("=" * width)
    print(title)
    print("=" * width)


def print_separator(width: int = 80) -> None:
    """Print a separator line."""
    print("-" * width)


# =============================================================================
# Figure Saving
# =============================================================================

def savefig(
    fig_fn,
    prefix: str,
    **kwargs,
) -> None:
    """
    Save figure to SVG and PNG in both light and dark themes.

    This function generates git-friendly SVG files (no Date metadata) with
    transparent backgrounds, plus PNG versions for easy viewing, in both
    light and dark color schemes for use in blog posts that support theme switching.

    Args:
        fig_fn: A callable that returns a matplotlib Figure. Will be called
            twice - once for each theme.
        prefix: Filename prefix. Files will be saved as:
            - figures/{prefix}-light.svg
            - figures/{prefix}-light.png
            - figures/{prefix}-dark.svg
            - figures/{prefix}-dark.png
        **kwargs: Additional arguments passed to fig.savefig().
    """
    light_svg_path = FIGURES_DIR / f"{prefix}-light.svg"
    light_png_path = FIGURES_DIR / f"{prefix}-light.png"
    dark_svg_path = FIGURES_DIR / f"{prefix}-dark.svg"
    dark_png_path = FIGURES_DIR / f"{prefix}-dark.png"

    # Generate light theme version
    with plt.style.context("default"):
        fig = fig_fn()
        fig.savefig(light_svg_path, metadata={"Date": None}, transparent=True, bbox_inches="tight", **kwargs)
        fig.savefig(light_png_path, dpi=150, transparent=True, bbox_inches="tight", **kwargs)
        plt.close(fig)

    # Generate dark theme version
    with plt.style.context("dark_background"):
        fig = fig_fn()
        fig.savefig(dark_svg_path, metadata={"Date": None}, transparent=True, bbox_inches="tight", **kwargs)
        fig.savefig(dark_png_path, dpi=150, transparent=True, bbox_inches="tight", **kwargs)
        plt.close(fig)

    print(f"Saved: {light_svg_path}, {light_png_path}")
    print(f"Saved: {dark_svg_path}, {dark_png_path}")


# =============================================================================
# SECTION 1: ENERGY PER RESPONSE
# =============================================================================

# =============================================================================
# Section 1.1: More Tokens, More Energy (LLMs)
# =============================================================================

def section1_1_llm_tokens() -> None:
    """
    Analyze how output tokens determine energy per response for LLMs.

    Generates violin plots comparing Problem Solving vs Text Conversation tasks
    across energy per token, output length, and energy per response.
    """
    print_header("SECTION 1.1: More Tokens, More Energy (LLMs)")

    gpqa_configs = get_configs("gpqa")
    chat_configs = get_configs("lm-arena-chat")

    # Get max batch config per model on B200
    def get_min_energy_configs(configs: list[dict]) -> dict[str, dict]:
        by_model: dict[str, dict] = {}
        for c in configs:
            if c["gpu_model"] != "B200":
                continue
            model = c["nickname"]
            if model not in by_model or c["max_num_seqs"] > by_model[model]["max_num_seqs"]:
                by_model[model] = c
        return by_model

    gpqa_by_model = get_min_energy_configs(gpqa_configs)
    chat_by_model = get_min_energy_configs(chat_configs)

    # Token stats: mean output tokens per task
    gpqa_tokens = [require(c, "avg_output_len") for c in gpqa_by_model.values()]
    chat_tokens = [require(c, "avg_output_len") for c in chat_by_model.values()]
    gpqa_tokens_mean = float(np.mean(gpqa_tokens))
    chat_tokens_mean = float(np.mean(chat_tokens))
    tokens_ratio = gpqa_tokens_mean / chat_tokens_mean

    print(f"\nOutput tokens: Problem Solving mean {gpqa_tokens_mean:,.0f} vs Text Conversation mean {chat_tokens_mean:,.0f} = {tokens_ratio:.0f}x more")

    # Energy per response stats
    gpqa_energy = [require(c, "energy_per_request_joules") for c in gpqa_by_model.values()]
    chat_energy = [require(c, "energy_per_request_joules") for c in chat_by_model.values()]
    gpqa_energy_mean = float(np.mean(gpqa_energy))
    chat_energy_mean = float(np.mean(chat_energy))
    energy_ratio = gpqa_energy_mean / chat_energy_mean

    print(f"Energy/response: Problem Solving mean {gpqa_energy_mean:,.0f} J vs Text Conversation mean {chat_energy_mean:,.0f} J = {energy_ratio:.0f}x more")

    # Case study: Qwen 3 32B on B200 x1
    print("\n--- Case Study: Qwen 3 32B on B200 x1 ---")
    model = "Qwen 3 32B"
    gpqa_32b = filter_configs(gpqa_configs, nickname=model, gpu_model="B200", num_gpus=1)
    chat_32b = filter_configs(chat_configs, nickname=model, gpu_model="B200", num_gpus=1)

    gpqa_max = max(gpqa_32b, key=lambda c: c["max_num_seqs"])
    chat_max = max(chat_32b, key=lambda c: c["max_num_seqs"])

    print(f"{'Metric':<30} {'Problem Solving':>18} {'Text Conversation':>18} {'Ratio':>12}")
    print_separator(80)
    print(f"{'Max batch size':<30} {gpqa_max['max_num_seqs']:>18,} {chat_max['max_num_seqs']:>18,} {chat_max['max_num_seqs']/gpqa_max['max_num_seqs']:>10.0f}x lower")
    print(f"{'Average output tokens':<30} {require(gpqa_max, 'avg_output_len'):>18,.0f} {require(chat_max, 'avg_output_len'):>18,.0f} {require(gpqa_max, 'avg_output_len')/require(chat_max, 'avg_output_len'):>10.0f}x more")
    print(f"{'Energy/token @ max batch (J)':<30} {gpqa_max['energy_per_token_joules']:>18.3f} {chat_max['energy_per_token_joules']:>18.3f} {gpqa_max['energy_per_token_joules']/chat_max['energy_per_token_joules']:>10.1f}x higher")

    # Same batch comparison
    same_batch = gpqa_max["max_num_seqs"]
    chat_at_batch = next((c for c in chat_32b if c["max_num_seqs"] == same_batch), None)
    if chat_at_batch:
        print(f"{'Energy/token @ batch ' + str(same_batch) + ' (J)':<30} {gpqa_max['energy_per_token_joules']:>18.3f} {chat_at_batch['energy_per_token_joules']:>18.3f} {gpqa_max['energy_per_token_joules']/chat_at_batch['energy_per_token_joules']:>10.1f}x higher")

    print(f"{'Energy/response (J)':<30} {gpqa_max['energy_per_request_joules']:>18,.0f} {chat_max['energy_per_request_joules']:>18,.0f} {gpqa_max['energy_per_request_joules']/chat_max['energy_per_request_joules']:>10.0f}x more")

    def draw():
        fig, axes = plt.subplots(1, 3, figsize=(12, 5), tight_layout=True)

        # Collect per-model data for Problem Solving and Text Conversation
        # Use max batch config per model (best-case energy)
        gpqa_configs = get_configs("gpqa")
        chat_configs = get_configs("lm-arena-chat")

        # Group by model, get max batch config for each
        def get_best_configs(configs):
            by_model = {}
            for c in configs:
                if c["gpu_model"] != "B200":
                    continue
                model = c["nickname"]
                if model not in by_model or c["max_num_seqs"] > by_model[model]["max_num_seqs"]:
                    by_model[model] = c
            return list(by_model.values())

        gpqa_best = get_best_configs(gpqa_configs)
        chat_best = get_best_configs(chat_configs)

        # Extract data
        gpqa_energy_tok = [c["energy_per_token_joules"] for c in gpqa_best]
        chat_energy_tok = [c["energy_per_token_joules"] for c in chat_best]
        gpqa_tokens = [require(c, "avg_output_len") for c in gpqa_best]
        chat_tokens = [require(c, "avg_output_len") for c in chat_best]
        gpqa_energy = [c["energy_per_request_joules"] for c in gpqa_best]
        chat_energy = [c["energy_per_request_joules"] for c in chat_best]

        # Common violin plot settings
        positions = [1, 1.7]  # Closer together
        width = 0.6  # Narrower violins

        def style_violin(parts):
            parts["bodies"][0].set_facecolor(COLORS["red"])
            parts["bodies"][1].set_facecolor(COLORS["blue"])
            for pc in parts["bodies"]:
                pc.set_zorder(3)

        # Panel 1: Energy per token
        ax1 = axes[0]
        parts1 = ax1.violinplot([gpqa_energy_tok, chat_energy_tok], positions=positions, widths=width, showmedians=True)
        style_violin(parts1)
        ax1.set_xticks(positions)
        ax1.set_xticklabels(["Problem\nSolving", "Text\nConversation"], fontsize=12)
        ax1.set_ylabel("Energy per Token (J)", fontsize=13)
        ax1.set_title("Energy per Token", fontsize=14)
        ax1.tick_params(axis='y', labelsize=11)
        ax1.set_xlim(0.5, 2.2)
        ax1.set_ylim(0, None)

        # Panel 2: Output tokens
        ax2 = axes[1]
        parts2 = ax2.violinplot([gpqa_tokens, chat_tokens], positions=positions, widths=width, showmedians=True)
        style_violin(parts2)
        ax2.set_xticks(positions)
        ax2.set_xticklabels(["Problem\nSolving", "Text\nConversation"], fontsize=12)
        ax2.set_ylabel("Output Tokens", fontsize=13)
        ax2.set_title("Output Length", fontsize=14)
        ax2.tick_params(axis='y', labelsize=11)
        ax2.set_xlim(0.5, 2.2)
        ax2.set_ylim(0, None)

        # Panel 3: Energy per response
        ax3 = axes[2]
        parts3 = ax3.violinplot([gpqa_energy, chat_energy], positions=positions, widths=width, showmedians=True)
        style_violin(parts3)
        ax3.set_xticks(positions)
        ax3.set_xticklabels(["Problem\nSolving", "Text\nConversation"], fontsize=12)
        ax3.set_ylabel("Energy per Response (J)", fontsize=13)
        ax3.set_title("Energy per Response", fontsize=14)
        ax3.tick_params(axis='y', labelsize=11)
        ax3.set_xlim(0.5, 2.2)
        ax3.set_ylim(0, None)

        return fig

    savefig(draw, "section1-1-llm-tokens")


# =============================================================================
# Section 1.2: Multimodal LLMs - Image and Video Understanding
# =============================================================================

def section1_2_mllm() -> Section1_2Result:
    """
    Analyze energy consumption of multimodal LLMs for image and video understanding.

    Key finding: Multimodal responses cost 2-4x more energy than text due to
    batch size limitations from preprocessing overhead, not just output length.
    """
    print_header("SECTION 1.2: Multimodal LLMs - Image and Video Understanding")

    text_chat = get_configs("lm-arena-chat")
    image_chat = get_configs("image-chat")
    video_chat = get_configs("video-chat")

    # Map VLM models to their text counterparts
    vlm_to_text_map = {
        "Qwen 3 VL 8B Instruct": "Qwen 3 8B",
        "Qwen 3 VL 32B Instruct": "Qwen 3 32B",
        "Qwen 3 VL 30B A3B Instruct": "Qwen 3 30B A3B Instruct",
        "Qwen 3 Omni 30B A3B Instruct": "Qwen 3 30B A3B Instruct",
        "Qwen 3 VL 235B A22B Instruct": "Qwen 3 235B A22B Instruct",
    }

    # Get min energy per token config for each model/modality on B200
    def get_min_energy_config(configs: list[dict], model: str) -> dict | None:
        filtered = [c for c in configs if c["nickname"] == model and c["gpu_model"] == "B200"]
        if not filtered:
            filtered = [c for c in configs if c["nickname"] == model]
        return min(filtered, key=lambda c: c["energy_per_token_joules"]) if filtered else None

    # Case study: Qwen 3 VL 8B vs Qwen 3 8B
    print("\n--- Case Study: Qwen 3 VL 8B vs Qwen 3 8B on B200 ---\n")
    vlm_model = "Qwen 3 VL 8B Instruct"
    text_model = vlm_to_text_map[vlm_model]

    txt_cfg = get_min_energy_config(text_chat, text_model)
    img_cfg = get_min_energy_config(image_chat, vlm_model)
    vid_cfg = get_min_energy_config(video_chat, vlm_model)

    if txt_cfg and img_cfg and vid_cfg:
        print(f"{'Metric':<30} {'Text':<15} {'Image':<15} {'Video':<15}")
        print_separator(75)
        print(f"{'Max batch size':<30} {txt_cfg['max_num_seqs']:<15} {img_cfg['max_num_seqs']:<15} {vid_cfg['max_num_seqs']:<15}")
        print(f"{'Energy/tok @ max batch (J)':<30} {txt_cfg['energy_per_token_joules']:<15.4f} {img_cfg['energy_per_token_joules']:<15.4f} {vid_cfg['energy_per_token_joules']:<15.4f}")
        print(f"{'Avg output tokens':<30} {txt_cfg['avg_output_len']:<15.0f} {img_cfg['avg_output_len']:<15.0f} {vid_cfg['avg_output_len']:<15.0f}")
        print(f"{'Energy/response (J)':<30} {txt_cfg['energy_per_request_joules']:<15.0f} {img_cfg['energy_per_request_joules']:<15.0f} {vid_cfg['energy_per_request_joules']:<15.0f}")

        img_ratio = img_cfg["energy_per_request_joules"] / txt_cfg["energy_per_request_joules"]
        vid_ratio = vid_cfg["energy_per_request_joules"] / txt_cfg["energy_per_request_joules"]
        print(f"\nEnergy ratios (vs Text): Image {img_ratio:.1f}x, Video {vid_ratio:.1f}x")

        batch_reduction_img = txt_cfg["max_num_seqs"] / img_cfg["max_num_seqs"]
        batch_reduction_vid = txt_cfg["max_num_seqs"] / vid_cfg["max_num_seqs"]
        print(f"Batch size reduction: Image {batch_reduction_img:.0f}x, Video {batch_reduction_vid:.0f}x")

    # Collect results for all models
    text_results: list[MLLMResult] = []
    image_results: list[MLLMResult] = []
    video_results: list[MLLMResult] = []

    print("\n--- All VLM Models: Energy per Token (B200, max batch) ---\n")
    print(f"{'Model':<40} {'Text (J)':<12} {'Image (J)':<12} {'Video (J)':<12} {'I/T':<8} {'V/T':<8}")
    print_separator(92)

    vlm_models = sorted(set(c["nickname"] for c in image_chat) | set(c["nickname"] for c in video_chat))

    for vlm in vlm_models:
        img_cfg = get_min_energy_config(image_chat, vlm)
        vid_cfg = get_min_energy_config(video_chat, vlm)

        text_model = vlm_to_text_map.get(vlm, vlm)
        txt_cfg = get_min_energy_config(text_chat, text_model)

        if txt_cfg:
            text_results.append(MLLMResult(
                model=text_model,
                modality="text",
                energy_per_response=txt_cfg["energy_per_request_joules"],
                avg_output_len=txt_cfg["avg_output_len"],
                energy_per_token=txt_cfg["energy_per_token_joules"],
                batch_size=txt_cfg["max_num_seqs"],
            ))

        if img_cfg:
            image_results.append(MLLMResult(
                model=vlm,
                modality="image",
                energy_per_response=img_cfg["energy_per_request_joules"],
                avg_output_len=img_cfg["avg_output_len"],
                energy_per_token=img_cfg["energy_per_token_joules"],
                batch_size=img_cfg["max_num_seqs"],
            ))

        if vid_cfg:
            video_results.append(MLLMResult(
                model=vlm,
                modality="video",
                energy_per_response=vid_cfg["energy_per_request_joules"],
                avg_output_len=vid_cfg["avg_output_len"],
                energy_per_token=vid_cfg["energy_per_token_joules"],
                batch_size=vid_cfg["max_num_seqs"],
            ))

        txt_e = txt_cfg["energy_per_token_joules"] if txt_cfg else None
        img_e = img_cfg["energy_per_token_joules"] if img_cfg else None
        vid_e = vid_cfg["energy_per_token_joules"] if vid_cfg else None

        txt_str = f"{txt_e:.4f}" if txt_e else "-"
        img_str = f"{img_e:.4f}" if img_e else "-"
        vid_str = f"{vid_e:.4f}" if vid_e else "-"

        i_t = f"{img_e/txt_e:.1f}x" if (img_e and txt_e) else "-"
        v_t = f"{vid_e/txt_e:.1f}x" if (vid_e and txt_e) else "-"

        print(f"{vlm:<40} {txt_str:<12} {img_str:<12} {vid_str:<12} {i_t:<8} {v_t:<8}")

    # Generate plot
    def draw():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.set_axisbelow(True)
        ax2.set_axisbelow(True)

        # Panel 1: Energy per token by modality (bar chart for select models)
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
            txt_cfg = get_min_energy_config(text_chat, txt_model)
            img_cfg = get_min_energy_config(image_chat, vlm)
            vid_cfg = get_min_energy_config(video_chat, vlm)

            if txt_cfg and img_cfg:
                short_name = vlm.replace(" Instruct", "").replace(" 17B 16E", "")
                model_labels.append(short_name)
                text_energies.append(txt_cfg["energy_per_token_joules"])
                image_energies.append(img_cfg["energy_per_token_joules"])
                video_energies.append(vid_cfg["energy_per_token_joules"] if vid_cfg else 0)

        x = np.arange(len(model_labels))
        width = 0.25

        ax1.grid(True, axis="y", alpha=0.3)
        ax1.bar(x - width, text_energies, width, label="Text", color=COLORS["blue"])
        ax1.bar(x, image_energies, width, label="Image", color=COLORS["red"])
        ax1.bar(x + width, video_energies, width, label="Video", color=COLORS["green"])

        # Panel 2: Batch size and energy per token comparison (Qwen 3 VL 8B case study)
        vlm_model = "Qwen 3 VL 8B Instruct"
        text_model = "Qwen 3 8B"

        txt_cfg = get_min_energy_config(text_chat, text_model)
        img_cfg = get_min_energy_config(image_chat, vlm_model)
        vid_cfg = get_min_energy_config(video_chat, vlm_model)

        modalities = ["Text", "Image", "Video"]
        batch_sizes = [txt_cfg["max_num_seqs"], img_cfg["max_num_seqs"], vid_cfg["max_num_seqs"]]
        energy_per_tok = [txt_cfg["energy_per_token_joules"], img_cfg["energy_per_token_joules"], vid_cfg["energy_per_token_joules"]]

        # Shared Y axis range for energy per token (left panel Y, right panel Y2)
        energy_y_max = max(max(image_energies), max(video_energies), max(energy_per_tok)) * 1.15

        ax1.set_ylabel("Energy per Token (J)", fontsize=13)
        ax1.set_title("Energy per Token by Input Modality", fontsize=14)
        ax1.set_xticks(x)
        ax1.set_xticklabels(model_labels, rotation=30, ha="right", fontsize=10)
        ax1.tick_params(axis="y", labelsize=11)
        ax1.legend(fontsize=10)
        ax1.set_ylim(0, energy_y_max)

        ax2_twin = ax2.twinx()

        x2 = np.arange(len(modalities))
        width2 = 0.35

        ax2.grid(True, axis="y", alpha=0.3)
        bars1 = ax2.bar(x2 - width2/2, batch_sizes, width2, label="Batch Size", color=COLORS["blue"])
        bars2 = ax2_twin.bar(x2 + width2/2, energy_per_tok, width2, label="Energy/Token (J)", color=COLORS["red"])

        ax2.set_ylabel("Batch Size", fontsize=13, color=COLORS["blue"])
        ax2_twin.set_ylabel("Energy per Token (J)", fontsize=13, color=COLORS["red"])
        ax2.set_title("Qwen 3 (VL) 8B Case Study", fontsize=14)
        ax2.set_xticks(x2)
        ax2.set_xticklabels(modalities, fontsize=12)
        ax2.tick_params(axis="y", labelsize=11, colors=COLORS["blue"])
        ax2_twin.tick_params(axis="y", labelsize=11, colors=COLORS["red"])
        ax2.set_ylim(0, max(batch_sizes) * 1.2)
        ax2_twin.set_ylim(0, energy_y_max)

        # Combined legend
        # lines1, labels1 = ax2.get_legend_handles_labels()
        # lines2, labels2 = ax2_twin.get_legend_handles_labels()
        # ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper center", fontsize=10)

        fig.tight_layout()
        return fig

    savefig(draw, "section1-2-mllm")

    return Section1_2Result(
        text_results=text_results,
        image_results=image_results,
        video_results=video_results,
    )


# =============================================================================
# Section 1.3: Diffusion Models - Image and Video Generation
# =============================================================================

def section1_3_diffusion() -> Section1_3Result:
    """
    Analyze energy consumption of diffusion models (text-to-image, text-to-video).

    Key finding: Diffusion models have different energy drivers than LLMs:
    model size, denoising steps, resolution, and frame count.
    Energy per image varies 20x, and video uses 80-5000x more than images.
    """
    print_header("SECTION 1.3: Diffusion Models - Image and Video Generation")

    results = Section1_3Result(
        text_to_image=_analyze_text_to_image(),
        text_to_video=_analyze_text_to_video(),
    )

    # Generate plot
    def draw():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.set_axisbelow(True)
        ax2.set_axisbelow(True)

        # Panel 1: Text-to-image (vertical bars)
        t2i = results.text_to_image[:7]
        models1 = [format_model_with_params(r.model, r.params) for r in t2i]
        energies1 = [r.energy / 1000 for r in t2i]
        ax1.grid(True, axis='y', alpha=0.3)
        bars1 = ax1.bar(models1, energies1, color=COLORS["green"])
        ax1.set_ylabel("Energy per Image (kJ)", fontsize=13)
        ax1.set_title("Text-to-Image", fontsize=14)
        ax1.set_ylim(0, max(energies1) * 1.15)
        ax1.tick_params(axis='y', labelsize=11)
        plt.sca(ax1)
        plt.xticks(rotation=35, ha='right', fontsize=11)

        # Callout: Hunyuan-DiT 1.2 (1.5B) > SD 3.5 Large (8.1B) due to steps
        hunyuan_idx = next(i for i, r in enumerate(t2i) if "Hunyuan" in r.model)
        ax1.annotate(
            "1.5B model > 8.1B model\n(50 vs 28 steps)",
            xy=(hunyuan_idx, t2i[hunyuan_idx].energy / 1000),
            xytext=(hunyuan_idx - 1.5, t2i[hunyuan_idx].energy / 1000 + 0.8),
            fontsize=11,
            ha='center',
            arrowprops=dict(arrowstyle='->', color='gray', lw=1),
        )

        # Panel 2: Text-to-video (vertical bars)
        t2v = results.text_to_video
        models2 = [format_model_with_params(r.model, r.params) for r in t2v]
        energies2 = [r.energy / 1000 for r in t2v]
        ax2.grid(True, axis='y', alpha=0.3)
        bars2 = ax2.bar(models2, energies2, color=COLORS["red"])
        ax2.set_ylabel("Energy per Video (kJ)", fontsize=13)
        ax2.set_title("Text-to-Video", fontsize=14)
        ax2.set_ylim(0, max(energies2) * 1.15)
        ax2.tick_params(axis='y', labelsize=11)
        plt.sca(ax2)
        plt.xticks(rotation=35, ha='right', fontsize=11)

        # Add energy labels on top of the first two bars (too small to read otherwise)
        for i in range(2):
            ax2.text(i, energies2[i] + 50, f"{energies2[i]:.0f}", ha='center', fontsize=10)

        # Callout: CogVideoX 1.5 5B > Wan 2.1 14B due to resolution
        cog_idx = next(i for i, r in enumerate(t2v) if "CogVideoX 1.5" in r.model)
        ax2.annotate(
            "5B model > 14B model\n(768x1360 vs 480x832)",
            xy=(cog_idx, t2v[cog_idx].energy / 1000),
            xytext=(cog_idx - 1.2, t2v[cog_idx].energy / 1000 + 250),
            fontsize=11,
            ha='center',
            arrowprops=dict(arrowstyle='->', color='gray', lw=1),
        )

        fig.tight_layout()
        return fig

    savefig(draw, "section1-3-diffusion")

    return results


def _analyze_text_to_image() -> list[TextToImageResult]:
    """Analyze text-to-image energy per image."""
    print("\n--- Text-to-Image ---")

    t2i_configs = get_configs("text-to-image")

    # Filter to B200 only
    models: dict[str, list[dict]] = {}
    for c in t2i_configs:
        if c["gpu_model"] != "B200":
            continue
        if c["nickname"] not in models:
            models[c["nickname"]] = []
        models[c["nickname"]].append(c)

    results: list[TextToImageResult] = []
    for model, configs in models.items():
        best = get_best_config(configs, "energy_per_image_joules")
        if best:
            steps = best.get("inference_steps") or best.get("num_inference_steps")
            if steps is None:
                raise KeyError(f"Neither 'inference_steps' nor 'num_inference_steps' in config: {model}")
            results.append(TextToImageResult(
                model=model,
                params=require(best, "total_params_billions"),
                steps=steps,
                energy=require(best, "energy_per_image_joules"),
                throughput=require(best, "throughput_images_per_sec"),
            ))

    results = sorted(results, key=lambda x: x.energy)

    best_model = results[0]
    worst_model = results[-1]
    print(f"Energy range: {best_model.model} ({best_model.energy:.0f} J) to "
          f"{worst_model.model} ({worst_model.energy:.0f} J) = "
          f"{worst_model.energy/best_model.energy:.0f}x")

    return results


def _analyze_text_to_video() -> list[TextToVideoResult]:
    """Analyze text-to-video energy consumption."""
    print("\n--- Text-to-Video ---")

    t2v_configs = get_configs("text-to-video")

    # Filter to B200 only
    models: dict[str, list[dict]] = {}
    for c in t2v_configs:
        if c["gpu_model"] != "B200":
            continue
        if c["nickname"] not in models:
            models[c["nickname"]] = []
        models[c["nickname"]].append(c)

    results: list[TextToVideoResult] = []
    for model, configs in models.items():
        best = get_best_config(configs, "energy_per_video_joules")
        if best:
            results.append(TextToVideoResult(
                model=model,
                params=require(best, "total_params_billions"),
                resolution=f"{require(best, 'video_height')}x{require(best, 'video_width')}",
                frames=require(best, "num_frames"),
                energy=require(best, "energy_per_video_joules"),
                energy_wh=require(best, "energy_per_video_joules") / 3600,
            ))

    results = sorted(results, key=lambda x: x.energy)

    return results


# =============================================================================
# SECTION 2: ENERGY PER TOKEN
# =============================================================================

# =============================================================================
# Section 2.1: Batch Size Reduces Energy per Token
# =============================================================================

def section2_1_batch_size() -> tuple[list[dict], list[dict]]:
    """
    Analyze how batch size affects energy per token and throughput.

    Key finding: Increasing batch size reduces energy per token by 3-5x
    and increases throughput with diminishing returns.
    """
    print_header("SECTION 2.1: Batch Size - Energy and Throughput")

    # DeepSeek R1 example
    gpqa_configs = get_configs("gpqa")
    r1_configs = sorted(
        [c for c in filter_configs(gpqa_configs, nickname="DeepSeek R1", gpu_model="B200", num_gpus=8)
         if get_batch_size(c) >= MIN_BATCH_SIZE],
        key=lambda x: x["max_num_seqs"],
    )

    print("\nDeepSeek R1 on B200 x8:")
    print(f"{'Batch':<10} {'E/Token (J)':<14} {'Throughput (tok/s)':<18} {'Latency (ms)':<14}")
    print_separator(56)
    for c in r1_configs:
        print(f"{c['max_num_seqs']:<10} {c['energy_per_token_joules']:<14.4f} "
              f"{c['output_throughput_tokens_per_sec']:<18.0f} {c['median_itl_ms']:<14.1f}")

    if len(r1_configs) >= 2:
        e_reduction = r1_configs[0]["energy_per_token_joules"] / r1_configs[-1]["energy_per_token_joules"]
        t_increase = r1_configs[-1]["output_throughput_tokens_per_sec"] / r1_configs[0]["output_throughput_tokens_per_sec"]
        print(f"\nBatch {r1_configs[0]['max_num_seqs']} → {r1_configs[-1]['max_num_seqs']}: "
              f"{e_reduction:.1f}x energy reduction, {t_increase:.1f}x throughput increase")

    # Qwen 3 Coder example
    fim_configs = get_configs("sourcegraph-fim")
    coder_configs = sorted(
        [c for c in filter_configs(fim_configs, nickname="Qwen 3 Coder 30B A3B", gpu_model="B200", num_gpus=1)
         if get_batch_size(c) >= MIN_BATCH_SIZE],
        key=lambda x: x["max_num_seqs"],
    )

    print("\nQwen 3 Coder 30B A3B on B200 x1:")
    print(f"{'Batch':<10} {'E/Token (J)':<14} {'Throughput (tok/s)':<18} {'Latency (ms)':<14}")
    print_separator(56)
    for c in coder_configs[:8]:
        print(f"{c['max_num_seqs']:<10} {c['energy_per_token_joules']:<14.4f} "
              f"{c['output_throughput_tokens_per_sec']:<18.0f} {c['median_itl_ms']:<14.1f}")

    if len(coder_configs) >= 2:
        e_reduction = coder_configs[0]["energy_per_token_joules"] / coder_configs[-1]["energy_per_token_joules"]
        t_increase = coder_configs[-1]["output_throughput_tokens_per_sec"] / coder_configs[0]["output_throughput_tokens_per_sec"]
        print(f"\nBatch {coder_configs[0]['max_num_seqs']} → {coder_configs[-1]['max_num_seqs']}: "
              f"{e_reduction:.1f}x energy reduction, {t_increase:.1f}x throughput increase")

    def draw():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), tight_layout=True)
        ax1.set_axisbelow(True)
        ax2.set_axisbelow(True)

        batches = [c["max_num_seqs"] for c in r1_configs]
        energies = [c["energy_per_token_joules"] for c in r1_configs]
        throughputs = [c["output_throughput_tokens_per_sec"] for c in r1_configs]
        latencies = [c["median_itl_ms"] for c in r1_configs]
        powers = [c["avg_power_watts"] for c in r1_configs]

        # Normalize to percentage of max for comparison (power normalized to TDP: 1000W per B200)
        num_gpus_1 = r1_configs[0]["num_gpus"]
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
        ax1.set_title("DeepSeek R1 (Problem Solving) on 8x B200", fontsize=14)
        ax1.legend(fontsize=10, loc="lower center")
        ax1.set_ylim(0, 110)

        batches2 = [c["max_num_seqs"] for c in coder_configs]
        energies2 = [c["energy_per_token_joules"] for c in coder_configs]
        throughputs2 = [c["output_throughput_tokens_per_sec"] for c in coder_configs]
        latencies2 = [c["median_itl_ms"] for c in coder_configs]
        powers2 = [c["avg_power_watts"] for c in coder_configs]

        num_gpus_2 = coder_configs[0]["num_gpus"]
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
        ax2.set_title("Qwen 3 Coder 30B A3B (Code Completion) on 1x B200", fontsize=14)
        ax2.legend(fontsize=10, loc="center right")
        ax2.set_ylim(0, 110)

        return fig

    savefig(draw, "section2-1-batch-size")

    return r1_configs, coder_configs


# =============================================================================
# Section 2.2: MoE Architecture Achieves Lower Energy per Token
# =============================================================================

def section2_2_moe_energy() -> list[MoEEnergyResult]:
    """
    Compare MoE vs Dense architecture energy per token.

    Key finding: Energy per token scales with ACTIVE parameters, not total.
    MoE models achieve lower energy by activating only a fraction of parameters.

    Note: Only BF16 models for fair comparison.
    """
    print_header("SECTION 2.1: MoE vs Dense Architecture Energy per Token")

    gpqa_configs = get_configs("gpqa")

    # Include both MoE and Dense models across a range of sizes
    # All BF16 on B200 for fair comparison (no GPT OSS since it's not BF16)
    target_configs = [
        # Dense models
        ("Qwen 3 8B", "B200", 1),
        ("Qwen 3 14B", "B200", 1),
        ("Qwen 3 32B", "B200", 1),
        # MoE models
        ("Qwen 3 30B A3B Thinking", "B200", 1),
        ("Qwen 3 235B A22B Thinking", "B200", 8),
    ]

    results: list[MoEEnergyResult] = []
    for model, gpu, num_gpus in target_configs:
        configs = filter_configs(gpqa_configs, nickname=model, gpu_model=gpu, num_gpus=num_gpus, precision="bfloat16")
        best = get_best_config(configs, "energy_per_token_joules")
        if best:
            total_params = require(best, "total_params_billions")
            # activated_params_billions only exists for MoE models; dense models use all params
            active_params = best.get("activated_params_billions", total_params)
            arch = require(best, "architecture")
            is_moe = arch == "MoE"
            results.append(MoEEnergyResult(
                model=model + f"\n({num_gpus}x {gpu})",
                total_params=total_params,
                active_params=active_params,
                energy_per_token=require(best, "energy_per_token_joules"),
                architecture=arch,
                is_moe=is_moe,
            ))

    print(f"\n{'Model':<30} {'Total (B)':<12} {'Active (B)':<12} {'E/Token (J)':<14} {'Arch':<8}")
    print_separator(80)
    for r in sorted(results, key=lambda x: x.active_params):
        print(f"{r.model.replace('\n', ' '):<30} {r.total_params:<12.0f} {r.active_params:<12.0f} "
              f"{r.energy_per_token:<14.4f} {r.architecture:<8}")

    # Compare similar active params: MoE 3B active vs Dense 8B
    moe_3b = next((r for r in results if r.model.startswith("Qwen 3 30B A3B Thinking")), None)
    dense_32b = next((r for r in results if r.model.startswith("Qwen 3 32B")), None)
    if moe_3b and dense_32b:
        print(f"\nKey comparison:")
        print(f"  MoE 30B (3B active): {moe_3b.energy_per_token:.4f} J/tok")
        print(f"  Dense 32B (32B active): {dense_32b.energy_per_token:.4f} J/tok")
        print(f"  MoE achieves {dense_32b.energy_per_token / moe_3b.energy_per_token:.2f}x lower energy per token")
        print()

    def draw():
        fig, ax = plt.subplots(figsize=(6, 3.5), tight_layout=True)
        ax.set_axisbelow(True)

        # Manual label offsets to avoid overlap (points at x=3,8,14,22,32)
        label_offsets = {
            "Qwen 3 30B A3B": (10, -10),       # MoE 3B - right (one line)
            "Qwen 3 8B": (5, 10),            # Dense 8B - above right
            "Qwen 3 14B": (5, 10),         # Dense 14B - below right
            "Qwen 3 235B A22B": (5, 10),   # MoE 22B - left of dot
            "Qwen 3 32B": (5, 10),           # Dense 32B - right of dot
        }

        for r in results:
            color = COLORS["red"] if r.is_moe else COLORS["blue"]
            ax.scatter(r.active_params, r.energy_per_token, s=150, c=color,
                      edgecolors="black", linewidth=1)
            # Clean up model names: remove "Qwen 3 " prefix
            label = r.model.replace(" Thinking", "").replace(" Instruct", "")
            # For 30B A3B, put on one line to avoid x-axis overlap
            if r.model.startswith("Qwen 3 30B A3B"):
                label = label.replace("\n", " ")
            # Get offset for this model
            offset = (8, 5)  # default
            for model_prefix, off in label_offsets.items():
                if r.model.startswith(model_prefix):
                    offset = off
                    break
            ax.annotate(label, (r.active_params, r.energy_per_token),
                        xytext=offset, textcoords="offset points", fontsize=9)

        ax.set_xlabel("Active Parameters (Billions)", fontsize=13)
        ax.set_ylabel("Energy per Token (J)", fontsize=13)
        ax.set_title("Energy per Token by Active Parameters", fontsize=14)
        ax.tick_params(axis='both', labelsize=11)
        ax.set_xlim(0, 40)
        ax.set_ylim(0, 0.85)
        ax.grid(True, alpha=0.3)

        legend_elements = [
            Patch(facecolor=COLORS["red"], label="MoE"),
            Patch(facecolor=COLORS["blue"], label="Dense"),
        ]
        ax.legend(handles=legend_elements, loc="upper left", fontsize=11)

        return fig

    savefig(draw, "section2-1-moe-energy")

    return results


# =============================================================================
# Section 2.3: B200 vs H100 Energy Comparison (Iso-Time)
# =============================================================================

@dataclass
class IsoTimeComparison:
    """Result for iso-time GPU comparison."""
    model: str
    task: str
    deadline: float
    h100_energy: float
    h100_latency: float
    b200_energy: float
    b200_latency: float
    gain_pct: float
    winner: str


def section2_3_b200_vs_h100() -> list[IsoTimeComparison]:
    """
    Compare B200 vs H100 energy efficiency at matched latency deadlines.

    Key finding: B200 achieves lower energy than H100 in most cases,
    but H100 can win in specific regimes (e.g., high-latency video generation
    with fewer GPUs).
    """
    print_header("SECTION 2.2: B200 vs H100 Energy Comparison (Iso-Time)")

    all_comparisons: list[IsoTimeComparison] = []

    # Task display names
    task_display_names = {
        "gpqa": "Problem Solving",
        "lm-arena-chat": "Text Conversation",
        "sourcegraph-fim": "Code Infilling",
    }

    def find_best_config_for_deadline(
        configs: list[dict],
        model: str,
        gpu: str,
        precision: str,
        deadline: float,
        latency_key: str,
        energy_key: str,
    ) -> dict | None:
        """Find lowest-energy config that meets the latency deadline."""
        matching = [
            c for c in configs
            if c["nickname"] == model
            and c["gpu_model"] == gpu
            and c.get("weight_precision", "N/A") == precision
            and c[latency_key] <= deadline
        ]
        if not matching:
            return None
        return min(matching, key=lambda c: c[energy_key])

    # LLM comparisons (iso-ITL)
    print("\n--- LLM Models (Iso-ITL Comparison) ---")
    llm_comparisons: list[IsoTimeComparison] = []

    # Fixed ITL deadlines: 30ms (low), 50ms (medium), 100ms (high)
    llm_deadlines = [50, 100, 250]

    for task_id in ["gpqa", "lm-arena-chat"]:
        configs = get_configs(task_id)

        # Get unique (model, precision) pairs that have both H100 and B200 data
        model_precision_pairs: set[tuple[str, str]] = set()
        for c in configs:
            model_precision_pairs.add((c["nickname"], c.get("weight_precision", "N/A")))

        for model, precision in sorted(model_precision_pairs):
            model_prec_configs = [
                c for c in configs
                if c["nickname"] == model and c.get("weight_precision", "N/A") == precision
            ]
            h100_configs = [c for c in model_prec_configs if c["gpu_model"] == "H100"]
            b200_configs = [c for c in model_prec_configs if c["gpu_model"] == "B200"]

            if not h100_configs or not b200_configs:
                continue

            # Test at fixed ITL deadlines
            for deadline in llm_deadlines:
                h100 = find_best_config_for_deadline(
                    configs, model, "H100", precision, deadline, "median_itl_ms", "energy_per_token_joules"
                )
                b200 = find_best_config_for_deadline(
                    configs, model, "B200", precision, deadline, "median_itl_ms", "energy_per_token_joules"
                )

                if h100 and b200:
                    gain = (h100["energy_per_token_joules"] - b200["energy_per_token_joules"]) / h100["energy_per_token_joules"] * 100
                    llm_comparisons.append(IsoTimeComparison(
                        model=model,
                        task=task_id,
                        deadline=deadline,
                        h100_energy=h100["energy_per_token_joules"],
                        h100_latency=h100["median_itl_ms"],
                        b200_energy=b200["energy_per_token_joules"],
                        b200_latency=b200["median_itl_ms"],
                        gain_pct=gain,
                        winner="B200" if gain > 0 else "H100",
                    ))

    # Print LLM results
    b200_wins = sum(1 for c in llm_comparisons if c.winner == "B200")
    print(f"B200 wins {b200_wins}/{len(llm_comparisons)} iso-time comparisons")

    if llm_comparisons:
        gains = [c.gain_pct for c in llm_comparisons]
        print(f"B200 gain range: {min(gains):.1f}% to {max(gains):.1f}% (median {np.median(gains):.1f}%)")

    print(f"\n{'Model':<30} {'Task':<18} {'ITL (ms)':<10} {'H100 (J)':<10} {'B200 (J)':<10} {'Winner':<8}")
    print_separator(90)

    # Show representative comparisons
    shown = set()
    for c in sorted(llm_comparisons, key=lambda x: x.gain_pct):
        key = (c.model, c.task)
        if key not in shown and len(shown) < 8:
            shown.add(key)
            print(f"{c.model:<30} {c.task:<18} {c.deadline:<10.0f} {c.h100_energy:<10.4f} {c.b200_energy:<10.4f} {c.winner:<8}")

    # Diffusion comparisons (iso-latency)
    print("\n--- Diffusion Models (Iso-Latency Comparison) ---")
    image_comparisons: list[IsoTimeComparison] = []
    video_comparisons: list[IsoTimeComparison] = []

    # Fixed latency deadlines
    image_deadlines = [10, 30, 60]  # seconds
    video_deadlines = [100, 500, 1000]  # seconds

    for task_id, energy_key in [("text-to-image", "energy_per_image_joules"),
                                 ("text-to-video", "energy_per_video_joules")]:
        configs = get_configs(task_id)
        comparisons = image_comparisons if task_id == "text-to-image" else video_comparisons
        deadlines = image_deadlines if task_id == "text-to-image" else video_deadlines

        # Get unique (model, precision) pairs that have both H100 and B200 data
        model_precision_pairs: set[tuple[str, str]] = set()
        for c in configs:
            model_precision_pairs.add((c["nickname"], c.get("weight_precision", "N/A")))

        for model, precision in sorted(model_precision_pairs):
            model_prec_configs = [
                c for c in configs
                if c["nickname"] == model and c.get("weight_precision", "N/A") == precision
            ]
            h100_configs = [c for c in model_prec_configs if c["gpu_model"] == "H100"]
            b200_configs = [c for c in model_prec_configs if c["gpu_model"] == "B200"]

            if not h100_configs or not b200_configs:
                continue

            # Test at fixed latency deadlines
            for deadline in deadlines:
                h100 = find_best_config_for_deadline(
                    configs, model, "H100", precision, deadline, "batch_latency_s", energy_key
                )
                b200 = find_best_config_for_deadline(
                    configs, model, "B200", precision, deadline, "batch_latency_s", energy_key
                )

                if h100 and b200:
                    gain = (h100[energy_key] - b200[energy_key]) / h100[energy_key] * 100
                    comparisons.append(IsoTimeComparison(
                        model=model,
                        task=task_id,
                        deadline=deadline,
                        h100_energy=h100[energy_key],
                        h100_latency=h100["batch_latency_s"],
                        b200_energy=b200[energy_key],
                        b200_latency=b200["batch_latency_s"],
                        gain_pct=gain,
                        winner="B200" if gain > 0 else "H100",
                    ))

    # Print diffusion results
    for task_name, comparisons in [("Text-to-Image", image_comparisons), ("Text-to-Video", video_comparisons)]:
        if comparisons:
            b200_wins = sum(1 for c in comparisons if c.winner == "B200")
            gains = [c.gain_pct for c in comparisons]
            print(f"\n{task_name}: B200 wins {b200_wins}/{len(comparisons)}, gain range {min(gains):.1f}% to {max(gains):.1f}%")

    all_comparisons = llm_comparisons + image_comparisons + video_comparisons

    # Select N comparisons with equally spaced gains above a minimum threshold
    def get_spread_gains(comparisons: list[IsoTimeComparison], n: int = 4, min_gain: float = 0) -> list[IsoTimeComparison]:
        """Select N comparisons with equally spaced positive gains (high to low)."""
        # Filter to B200 wins above minimum gain, sort by gain descending
        b200_wins = [c for c in comparisons if c.gain_pct >= min_gain]
        if len(b200_wins) <= n:
            return sorted(b200_wins, key=lambda x: x.gain_pct, reverse=True)
        sorted_by_gain = sorted(b200_wins, key=lambda x: x.gain_pct, reverse=True)
        # Pick at evenly spaced indices: 0, 1/3, 2/3, 1 of the list
        indices = [int(i * (len(sorted_by_gain) - 1) / (n - 1)) for i in range(n)]
        return [sorted_by_gain[i] for i in indices]

    llm_plot_data = get_spread_gains(llm_comparisons, 4, min_gain=10)
    image_plot_data = get_spread_gains(image_comparisons, 4)
    video_plot_data = get_spread_gains(video_comparisons, 4)

    # Print cases where H100 wins
    h100_wins_llm = [c for c in llm_comparisons if c.winner == "H100"]
    h100_wins_image = [c for c in image_comparisons if c.winner == "H100"]
    h100_wins_video = [c for c in video_comparisons if c.winner == "H100"]

    if h100_wins_llm or h100_wins_image or h100_wins_video:
        print("\n--- Cases Where H100 Wins ---")
        for c in sorted(h100_wins_llm, key=lambda x: x.gain_pct):
            print(f"  LLM: {c.model} ({c.task}, ITL≤{c.deadline:.0f}ms): H100 {c.h100_energy:.4f}J vs B200 {c.b200_energy:.4f}J ({-c.gain_pct:.1f}% less energy)")
        for c in sorted(h100_wins_image, key=lambda x: x.gain_pct):
            print(f"  Image: {c.model} (deadline≤{c.deadline:.0f}s): H100 {c.h100_energy:.1f}J vs B200 {c.b200_energy:.1f}J ({-c.gain_pct:.1f}% less energy)")
        for c in sorted(h100_wins_video, key=lambda x: x.gain_pct):
            print(f"  Video: {c.model} (deadline≤{c.deadline:.0f}s): H100 {c.h100_energy:.1f}J vs B200 {c.b200_energy:.1f}J ({-c.gain_pct:.1f}% less energy)")

    # Fine-grained ITL scan for case studies
    def print_itl_scan(task_id: str, model_name: str, start: int = 15, end: int = 105, step: int = 5):
        """Print ITL scan comparing H100 vs B200 for a specific model."""
        configs = get_configs(task_id)
        h100_cs = [c for c in configs if c["nickname"] == model_name and c["gpu_model"] == "H100"]
        b200_cs = [c for c in configs if c["nickname"] == model_name and c["gpu_model"] == "B200"]

        def find_best(cfgs: list[dict], deadline: float) -> dict | None:
            matching = [c for c in cfgs if c["median_itl_ms"] <= deadline]
            return min(matching, key=lambda c: c["energy_per_token_joules"]) if matching else None

        print(f"\n--- Case Study: {model_name} ({task_id}) ITL Scan ---")
        print(f"{'Constraint':>10} | {'H100 (GPUs, batch, ITL, energy)':^35} | {'B200 (GPUs, batch, ITL, energy)':^35} | {'B200 Gain':>12}")
        print("-" * 100)
        for deadline in range(start, end, step):
            h100 = find_best(h100_cs, deadline)
            b200 = find_best(b200_cs, deadline)
            if h100 and b200:
                gain = (h100["energy_per_token_joules"] - b200["energy_per_token_joules"]) / h100["energy_per_token_joules"] * 100
                h_str = f"{h100['num_gpus']}x, b={h100['max_num_seqs']}, {h100['median_itl_ms']:.0f}ms, {h100['energy_per_token_joules']:.3f}J"
                b_str = f"{b200['num_gpus']}x, b={b200['max_num_seqs']}, {b200['median_itl_ms']:.0f}ms, {b200['energy_per_token_joules']:.3f}J"
                winner = "B200" if gain > 0 else "H100"
                print(f"{deadline:>10} | {h_str:^35} | {b_str:^35} | {gain:>+8.1f}% {winner}")
            elif h100:
                h_str = f"{h100['num_gpus']}x, b={h100['max_num_seqs']}, {h100['median_itl_ms']:.0f}ms, {h100['energy_per_token_joules']:.3f}J"
                print(f"{deadline:>10} | {h_str:^35} | {'(no config)':^35} | {'N/A':>12}")
            elif b200:
                b_str = f"{b200['num_gpus']}x, b={b200['max_num_seqs']}, {b200['median_itl_ms']:.0f}ms, {b200['energy_per_token_joules']:.3f}J"
                print(f"{deadline:>10} | {'(no config)':^35} | {b_str:^35} | {'N/A':>12}")

    # Case study 1: Qwen 3 30B A3B Thinking (H100 2x vs B200 1x)
    print_itl_scan("gpqa", "Qwen 3 30B A3B Thinking")

    # Case study 2: Qwen 3 235B A22B Instruct FP8 (H100 8x vs B200 2x)
    print_itl_scan("lm-arena-chat", "Qwen 3 235B A22B Instruct FP8", start=20, end=85, step=5)

    def draw():
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 5))
        ax1.set_axisbelow(True)
        ax2.set_axisbelow(True)
        ax3.set_axisbelow(True)

        width = 0.35

        # Left panel: LLM comparisons (with task names)
        models1 = [f"{c.model}\n{task_display_names.get(c.task, c.task)}" for c in llm_plot_data]
        h100_vals1 = [c.h100_energy for c in llm_plot_data]
        b200_vals1 = [c.b200_energy for c in llm_plot_data]

        x1 = np.arange(len(models1))

        ax1.grid(True, axis='y', alpha=0.3)
        ax1.bar(x1 - width/2, h100_vals1, width, label="H100", color=COLORS["red"])
        ax1.bar(x1 + width/2, b200_vals1, width, label="B200", color=COLORS["blue"])
        ax1.set_ylabel("Energy per Token (J)", fontsize=13)
        ax1.set_title(f"LLMs (Median ITL $\\leq$ {llm_deadlines[1]} ms)", fontsize=14)
        ax1.set_xticks(x1)
        ax1.set_xticklabels(models1, rotation=30, ha="right", fontsize=10)
        ax1.tick_params(axis='y', labelsize=11)
        ax1.legend(fontsize=10)
        ax1.set_ylim(0, max(h100_vals1) * 1.15)

        for i, c in enumerate(llm_plot_data):
            ax1.annotate(f"{c.gain_pct:.0f}%",
                        xy=(i, max(c.h100_energy, c.b200_energy)),
                        xytext=(0, 8), textcoords="offset points",
                        ha="center", fontsize=9)

        # Middle panel: Text-to-image comparisons
        if image_plot_data:
            models2 = [c.model for c in image_plot_data]
            h100_vals2 = [c.h100_energy for c in image_plot_data]
            b200_vals2 = [c.b200_energy for c in image_plot_data]

            x2 = np.arange(len(models2))

            ax2.grid(True, axis='y', alpha=0.3)
            ax2.bar(x2 - width/2, h100_vals2, width, label="H100", color=COLORS["red"])
            ax2.bar(x2 + width/2, b200_vals2, width, label="B200", color=COLORS["blue"])
            ax2.set_ylabel("Energy per Image (J)", fontsize=13)
            ax2.set_title(f"Text-to-Image (Gen. Time $\\leq$ {image_deadlines[1]} s)", fontsize=14)
            ax2.set_xticks(x2)
            ax2.set_xticklabels(models2, rotation=30, ha="right", fontsize=10)
            ax2.tick_params(axis='y', labelsize=11)
            ax2.legend(fontsize=10)
            ax2.set_ylim(0, max(h100_vals2) * 1.15)

            for i, c in enumerate(image_plot_data):
                ax2.annotate(f"{c.gain_pct:.0f}%",
                            xy=(i, max(c.h100_energy, c.b200_energy)),
                            xytext=(0, 8), textcoords="offset points",
                            ha="center", fontsize=9)

        # Right panel: Text-to-video comparisons
        if video_plot_data:
            models3 = [c.model for c in video_plot_data]
            h100_vals3 = [c.h100_energy for c in video_plot_data]
            b200_vals3 = [c.b200_energy for c in video_plot_data]

            x3 = np.arange(len(models3))

            ax3.grid(True, axis='y', alpha=0.3)
            ax3.bar(x3 - width/2, h100_vals3, width, label="H100", color=COLORS["red"])
            ax3.bar(x3 + width/2, b200_vals3, width, label="B200", color=COLORS["blue"])
            ax3.set_ylabel("Energy per Video (J)", fontsize=13)
            ax3.set_title(f"Text-to-Video (Gen. Time $\\leq$ {video_deadlines[1]} s)", fontsize=14)
            ax3.set_xticks(x3)
            ax3.set_xticklabels(models3, rotation=30, ha="right", fontsize=10)
            ax3.tick_params(axis='y', labelsize=11)
            ax3.legend(fontsize=10)
            ax3.set_ylim(0, max(h100_vals3) * 1.15)

            for i, c in enumerate(video_plot_data):
                ax3.annotate(f"{c.gain_pct:.0f}%",
                            xy=(i, max(c.h100_energy, c.b200_energy)),
                            xytext=(0, 8), textcoords="offset points",
                            ha="center", fontsize=9)

        fig.tight_layout()
        return fig

    savefig(draw, "section2-2-b200-vs-h100")

    return all_comparisons


# =============================================================================
# Section 2.4: Precision Affects Energy per Token
# =============================================================================

def section2_4_precision() -> None:
    """
    Analyze how precision (FP8, BF16) affects energy per token and latency.

    Key findings:
    1. FP8 typically wins at batch sizes ≥32 (common case)
    2. FP8 loses at low batch sizes due to GPU underutilization
    3. FP8 can be much worse when it requires attention DP (unusual case)
    """
    print_header("SECTION 2.4: Precision Affects Energy per Token")

    # FP8 vs BF16 analysis
    comparisons = _analyze_fp8_vs_bf16()

    # Typical case: Qwen 3 235B A22B Instruct on H100 x8 (FP8 wins at batch ≥32)
    typical_case = [c for c in comparisons
                    if "235B A22B Instruct" in c.model and "Thinking" not in c.model
                    and c.gpu == "H100" and c.num_gpus == 8 and c.task == "lm-arena-chat"]
    typical_sorted = sorted(typical_case, key=lambda x: x.batch)

    print("\nTypical case - Qwen 3 235B A22B Instruct on H100 x8:")
    print(f"{'Batch':<8} {'FP8 E(J)':<12} {'BF16 E(J)':<12} {'E Ratio':<10} {'FP8 L(ms)':<12} {'BF16 L(ms)':<12} {'L Ratio':<10}")
    print_separator(76)
    for c in typical_sorted:
        print(f"{c.batch:<8} {c.fp8_energy:<12.3f} {c.bf16_energy:<12.3f} {c.energy_ratio:<10.2f} {c.fp8_latency:<12.2f} {c.bf16_latency:<12.2f} {c.latency_ratio:<10.2f}")

    # Unusual case: 480B model (FP8 always loses due to attention DP)
    unusual_case = [c for c in comparisons if "480B" in c.model]
    unusual_sorted = sorted(unusual_case, key=lambda x: x.batch)

    print("\nUnusual case - Qwen 3 Coder 480B on B200 x8 (requires attention DP for FP8):")
    print(f"{'Batch':<8} {'FP8 E(J)':<12} {'BF16 E(J)':<12} {'E Ratio':<10}")
    print_separator(42)
    for c in unusual_sorted[:6]:
        print(f"{c.batch:<8} {c.fp8_energy:<12.3f} {c.bf16_energy:<12.3f} {c.energy_ratio:<10.2f}")

    # Calculate win rates per batch size for bar chart
    # Restrict to LLM tasks only, exclude 480B model (shown as separate case study)
    from collections import defaultdict
    llm_tasks = {"gpqa", "lm-arena-chat", "sourcegraph-fim"}
    batch_stats = defaultdict(lambda: {'e_wins': 0, 'l_wins': 0, 'total': 0})
    for c in comparisons:
        if c.task not in llm_tasks:
            continue  # Skip MLLM tasks
        if "480B" in c.model:
            continue  # Skip outlier case (shown separately)
        batch_stats[c.batch]['total'] += 1
        if c.fp8_wins_energy:
            batch_stats[c.batch]['e_wins'] += 1
        if c.fp8_wins_latency:
            batch_stats[c.batch]['l_wins'] += 1

    # Print win rates
    print("\nFP8 Win Rate by Batch Size:")
    print(f"{'Batch':<8} {'Total':<8} {'E Win%':<10} {'L Win%':<10}")
    print_separator(36)
    for batch in sorted(batch_stats.keys()):
        stats = batch_stats[batch]
        e_pct = 100 * stats['e_wins'] / stats['total']
        l_pct = 100 * stats['l_wins'] / stats['total']
        print(f"{batch:<8} {stats['total']:<8} {e_pct:<10.0f} {l_pct:<10.0f}")

    # Figure: Qwen 3 235B typical case study (two panels: Energy, Latency)
    def draw_case_study():
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), tight_layout=True)

        if typical_sorted:
            batches_t = [c.batch for c in typical_sorted]
            fp8_e = [c.fp8_energy for c in typical_sorted]
            bf16_e = [c.bf16_energy for c in typical_sorted]
            fp8_l = [c.fp8_latency for c in typical_sorted]
            bf16_l = [c.bf16_latency for c in typical_sorted]

            # Panel 1: Energy
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

            # Panel 2: Latency
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

    savefig(draw_case_study, "section2-4-precision")


def _analyze_fp8_vs_bf16() -> list[FP8BF16Comparison]:
    """Analyze FP8 vs BF16 quantization energy per token and latency."""
    print("\n--- FP8 vs BF16 ---")

    comparisons: list[FP8BF16Comparison] = []

    for task_id, task_data in TASKS.items():
        configs = task_data.get("configurations", [])

        groups: dict[tuple, dict[str, dict]] = {}
        for c in configs:
            base_name = c["nickname"].replace(" FP8", "").replace(" fp8", "")
            batch = get_batch_size(c)
            key = (base_name, c["gpu_model"], c["num_gpus"], batch)
            if key not in groups:
                groups[key] = {}
            groups[key][c["weight_precision"]] = c

        for key, prec_configs in groups.items():
            if "fp8" in prec_configs and "bfloat16" in prec_configs:
                fp8 = prec_configs["fp8"]
                bf16 = prec_configs["bfloat16"]
                energy_key = "energy_per_token_joules" if "energy_per_token_joules" in fp8 else "energy_per_image_joules"
                latency_key = "median_itl_ms" if "median_itl_ms" in fp8 else "batch_latency_s"

                fp8_energy = fp8[energy_key]
                bf16_energy = bf16[energy_key]
                energy_ratio = fp8_energy / bf16_energy

                fp8_latency = fp8.get(latency_key, 0)
                bf16_latency = bf16.get(latency_key, 0)
                latency_ratio = fp8_latency / bf16_latency if bf16_latency > 0 else 1.0

                comparisons.append(FP8BF16Comparison(
                    task=task_id,
                    model=key[0],
                    gpu=key[1],
                    num_gpus=key[2],
                    batch=key[3],
                    fp8_energy=fp8_energy,
                    bf16_energy=bf16_energy,
                    energy_ratio=energy_ratio,
                    fp8_wins_energy=energy_ratio < 1,
                    fp8_latency=fp8_latency,
                    bf16_latency=bf16_latency,
                    latency_ratio=latency_ratio,
                    fp8_wins_latency=latency_ratio < 1,
                ))

    # Print summary by batch size range
    batch_ranges = [(1, 16, "1-16"), (17, 64, "17-64"), (65, 256, "65-256"), (257, 1024, "257-1024")]
    print("\nFP8 win rate by batch size range:")
    print(f"{'Batch Range':<15} {'Energy Win %':<15} {'Latency Win %':<15}")
    print_separator(45)
    for low, high, label in batch_ranges:
        in_range = [c for c in comparisons if low <= c.batch <= high]
        if in_range:
            e_wins = sum(1 for c in in_range if c.fp8_wins_energy)
            l_wins = sum(1 for c in in_range if c.fp8_wins_latency)
            total = len(in_range)
            print(f"{label:<15} {100*e_wins/total:>5.0f}% ({e_wins}/{total}){'':<3} {100*l_wins/total:>5.0f}% ({l_wins}/{total})")

    return comparisons


# =============================================================================
# Section 2.5: Multi-GPU Scaling
# =============================================================================

def section2_5_multi_gpu_scaling() -> None:
    """
    Analyze how multi-GPU scaling affects energy per token and latency.

    Key findings:
    1. At same batch, more GPUs reduce latency but INCREASE energy (100% on B200, >90% on H100)
    2. Multi-GPU is primarily a latency optimization
    3. Energy benefit only when model requires more GPUs to fit (capacity unlock)
    """
    print_header("SECTION 2.5: Multi-GPU Scaling")

    # Use GPT OSS 120B on B200 - shows 100% pattern (E up, L down)
    configs = get_configs("gpqa")
    model_configs = [c for c in configs
                     if c["nickname"] == "GPT OSS 120B"
                     and c["gpu_model"] == "B200"]

    gpu1_configs = sorted([c for c in model_configs if c["num_gpus"] == 1 and c["max_num_seqs"] >= 16],
                          key=lambda x: x["max_num_seqs"])
    gpu2_configs = sorted([c for c in model_configs if c["num_gpus"] == 2 and c["max_num_seqs"] >= 16],
                          key=lambda x: x["max_num_seqs"])

    print("\n--- GPT OSS 120B on B200: Time-Energy Tradeoff ---")
    print(f"{'GPUs':<6} {'Batch':<8} {'Energy':<12} {'Latency':<12}")
    print("-" * 40)
    for c in gpu1_configs:
        print(f"{c['num_gpus']:<6} {c['max_num_seqs']:<8} {c['energy_per_token_joules']:<12.3f} {c['median_itl_ms']:<12.2f}")
    for c in gpu2_configs:
        print(f"{c['num_gpus']:<6} {c['max_num_seqs']:<8} {c['energy_per_token_joules']:<12.3f} {c['median_itl_ms']:<12.2f}")

    # Verification: At same batch, what happens to energy vs latency?
    # Analyze separately for H100 and B200
    print("\n--- Verification: Same batch, more GPUs (by GPU type) ---")
    from collections import defaultdict

    # Load all LLM tasks for comprehensive analysis
    all_configs = []
    for task_id in ["gpqa", "lm-arena-chat", "sourcegraph-fim"]:
        all_configs.extend(get_configs(task_id))

    model_groups = defaultdict(lambda: defaultdict(list))
    for c in all_configs:
        key = (c["nickname"], c["gpu_model"])
        model_groups[key][c["num_gpus"]].append(c)

    for gpu_type in ["B200", "H100"]:
        e_up_l_down = 0
        e_down_l_down = 0
        e_up_l_up = 0
        e_down_l_up = 0
        total = 0

        for (nickname, gpu), gpu_configs in sorted(model_groups.items()):
            if gpu != gpu_type:
                continue
            # Compare all pairs of GPU counts where N < M
            gpu_counts = sorted(gpu_configs.keys())
            for i, n_gpus in enumerate(gpu_counts):
                for m_gpus in gpu_counts[i+1:]:
                    batches_n = {c["max_num_seqs"]: c for c in gpu_configs[n_gpus]}
                    batches_m = {c["max_num_seqs"]: c for c in gpu_configs[m_gpus]}
                    common_batches = set(batches_n.keys()) & set(batches_m.keys())
                    for batch in common_batches:
                        cn = batches_n[batch]
                        cm = batches_m[batch]
                        total += 1
                        e_down = cm["energy_per_token_joules"] < cn["energy_per_token_joules"]
                        l_down = cm["median_itl_ms"] < cn["median_itl_ms"]
                        if e_down and l_down:
                            e_down_l_down += 1
                        elif not e_down and l_down:
                            e_up_l_down += 1
                        elif not e_down and not l_down:
                            e_up_l_up += 1
                        else:
                            e_down_l_up += 1

        if total > 0:
            print(f"\n  {gpu_type}:")
            print(f"    E up, L down: {e_up_l_down}/{total} ({100*e_up_l_down/total:.0f}%)")
            print(f"    E down, L down: {e_down_l_down}/{total} ({100*e_down_l_down/total:.0f}%)")
            print(f"    E up, L up: {e_up_l_up}/{total} ({100*e_up_l_up/total:.0f}%)")
            print(f"    E down, L up: {e_down_l_up}/{total} ({100*e_down_l_up/total:.0f}%)")
            print(f"    => Latency decreases: {e_up_l_down + e_down_l_down}/{total} ({100*(e_up_l_down + e_down_l_down)/total:.0f}%)")
            print(f"    => Energy increases: {e_up_l_down + e_up_l_up}/{total} ({100*(e_up_l_down + e_up_l_up)/total:.0f}%)")

    # Verification: capacity unlock (by GPU type)
    print("\n--- Verification: Capacity Unlock (by GPU type) ---")
    print("  (Capacity unlock = more GPUs enable larger max batch AND lower min energy)")
    for gpu_type in ["B200", "H100"]:
        valid_capacity = 0
        total_capacity = 0
        details = []
        for (nickname, gpu), gpu_configs in sorted(model_groups.items()):
            if gpu != gpu_type:
                continue
            # Compare all pairs of GPU counts where N < M
            gpu_counts = sorted(gpu_configs.keys())
            for i, n_gpus in enumerate(gpu_counts):
                for m_gpus in gpu_counts[i+1:]:
                    min_e_n = min(c["energy_per_token_joules"] for c in gpu_configs[n_gpus])
                    min_e_m = min(c["energy_per_token_joules"] for c in gpu_configs[m_gpus])
                    max_b_n = max(c["max_num_seqs"] for c in gpu_configs[n_gpus])
                    max_b_m = max(c["max_num_seqs"] for c in gpu_configs[m_gpus])
                    total_capacity += 1
                    unlocked = min_e_m < min_e_n and max_b_m > max_b_n
                    if unlocked:
                        valid_capacity += 1
                    details.append((nickname, n_gpus, max_b_n, min_e_n, m_gpus, max_b_m, min_e_m, unlocked))
        if total_capacity > 0:
            print(f"\n  {gpu_type}: Capacity unlock: {valid_capacity}/{total_capacity} ({100*valid_capacity/total_capacity:.0f}%)")
            for nickname, n_gpus, max_b_n, min_e_n, m_gpus, max_b_m, min_e_m, unlocked in details:
                status = "✓" if unlocked else "✗"
                print(f"    {status} {nickname}: {n_gpus}GPU(batch={max_b_n}, E={min_e_n:.3f}) -> {m_gpus}GPU(batch={max_b_m}, E={min_e_m:.3f})")

    # Also get H100 configs for GPT OSS 120B
    h100_model_configs = [c for c in all_configs
                          if c["nickname"] == "GPT OSS 120B"
                          and c["gpu_model"] == "H100"]
    h100_gpu1_configs = sorted([c for c in h100_model_configs if c["num_gpus"] == 1 and c["max_num_seqs"] >= 16],
                               key=lambda x: x["max_num_seqs"])
    h100_gpu2_configs = sorted([c for c in h100_model_configs if c["num_gpus"] == 2 and c["max_num_seqs"] >= 16],
                               key=lambda x: x["max_num_seqs"])

    def draw():
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), tight_layout=True)

        # Left panel: B200 (no capacity unlock benefit)
        ax1 = axes[0]
        ax1.set_axisbelow(True)
        ax1.grid(True, alpha=0.3)

        lat1 = [c["median_itl_ms"] for c in gpu1_configs]
        eng1 = [c["energy_per_token_joules"] for c in gpu1_configs]
        batch1 = [c["max_num_seqs"] for c in gpu1_configs]
        ax1.plot(lat1, eng1, "o-", color=COLORS["blue"], linewidth=2, markersize=10, label="1 GPU", zorder=3)

        lat2 = [c["median_itl_ms"] for c in gpu2_configs]
        eng2 = [c["energy_per_token_joules"] for c in gpu2_configs]
        batch2 = [c["max_num_seqs"] for c in gpu2_configs]
        ax1.plot(lat2, eng2, "s-", color=COLORS["red"], linewidth=2, markersize=10, label="2 GPUs", zorder=3)

        for i, (x, y, b) in enumerate(zip(lat1, eng1, batch1)):
            if b in [16, 64, 512, 2048, 3072]:
                ax1.annotate(str(b), (x, y), textcoords="offset points", xytext=(5, 5), fontsize=9)
        for i, (x, y, b) in enumerate(zip(lat2, eng2, batch2)):
            if b in [16, 64, 512, 2048, 4096]:
                ax1.annotate(str(b), (x, y), textcoords="offset points", xytext=(5, 5), fontsize=9)

        ax1.set_xlabel("Median ITL (ms)", fontsize=13)
        ax1.set_ylabel("Energy per Token (J)", fontsize=13)
        ax1.set_title("GPT OSS 120B (Problem Solving) on B200", fontsize=14)
        ax1.tick_params(axis="both", labelsize=11)
        ax1.legend(fontsize=11, loc="upper right")
        ax1.set_xlim(left=0)
        ax1.set_ylim(bottom=0)

        # Right panel: H100 (capacity unlock benefit)
        ax2 = axes[1]
        ax2.set_axisbelow(True)
        ax2.grid(True, alpha=0.3)

        if h100_gpu1_configs and h100_gpu2_configs:
            lat1_h = [c["median_itl_ms"] for c in h100_gpu1_configs]
            eng1_h = [c["energy_per_token_joules"] for c in h100_gpu1_configs]
            batch1_h = [c["max_num_seqs"] for c in h100_gpu1_configs]
            ax2.plot(lat1_h, eng1_h, "o-", color=COLORS["blue"], linewidth=2, markersize=10, label="1 GPU", zorder=3)

            lat2_h = [c["median_itl_ms"] for c in h100_gpu2_configs]
            eng2_h = [c["energy_per_token_joules"] for c in h100_gpu2_configs]
            batch2_h = [c["max_num_seqs"] for c in h100_gpu2_configs]
            ax2.plot(lat2_h, eng2_h, "s-", color=COLORS["red"], linewidth=2, markersize=10, label="2 GPUs", zorder=3)

            for i, (x, y, b) in enumerate(zip(lat1_h, eng1_h, batch1_h)):
                if b in [16, 64]:
                    ax2.annotate(str(b), (x, y), textcoords="offset points", xytext=(5, 5), fontsize=9)
            for i, (x, y, b) in enumerate(zip(lat2_h, eng2_h, batch2_h)):
                if b in [16, 64, 512, 2048]:
                    ax2.annotate(str(b), (x, y), textcoords="offset points", xytext=(5, 5), fontsize=9)

        ax2.set_xlabel("Median ITL (ms)", fontsize=13)
        ax2.set_ylabel("Energy per Token (J)", fontsize=13)
        ax2.set_title("GPT OSS 120B (Problem Solving) on H100", fontsize=14)
        ax2.tick_params(axis="both", labelsize=11)
        ax2.legend(fontsize=11, loc="upper right")
        ax2.set_xlim(left=0)
        ax2.set_ylim(bottom=0)

        return fig

    savefig(draw, "section2-5-multi-gpu")


# =============================================================================
# SECTION 3: POWER
# =============================================================================

# =============================================================================
# Section 3.1: Model Size Affects Power
# =============================================================================

def section3_1_model_size_power() -> list[ModelPowerResult]:
    """
    Analyze how model size affects GPU power consumption.

    Key finding: Larger models draw more power, but the relationship is not
    linear. MoE models draw less power than dense models of similar total size.
    """
    print_header("SECTION 3.1: Model Size Affects Power")

    chat_configs = get_configs("lm-arena-chat")

    # Get all models that can run on 1 GPU for each GPU type
    all_results: dict[str, list[ModelPowerResult]] = {"B200": [], "H100": []}

    for gpu_type in ["B200", "H100"]:
        tdp = 1000 if gpu_type == "B200" else 700
        print(f"\n--- {gpu_type} (TDP: {tdp}W) ---")
        print(f"{'Model':<35} {'Active':<8} {'Power':<8} {'Batch':<8}")
        print_separator(60)

        # Get all unique models on this GPU with 1 GPU
        gpu_configs = filter_configs(chat_configs, gpu_model=gpu_type, num_gpus=1)
        models = sorted(set(c["nickname"] for c in gpu_configs))

        for model in models:
            configs = filter_configs(chat_configs, nickname=model, gpu_model=gpu_type, num_gpus=1)
            if configs:
                # Get max batch size config
                best = max(configs, key=lambda c: c["max_num_seqs"])

                if best:
                    total_params = require(best, "total_params_billions")
                    active_params = best.get("activated_params_billions", total_params)
                    batch = get_batch_size(best)

                    all_results[gpu_type].append(ModelPowerResult(
                        model=model,
                        active_params=active_params,
                        gpu_power=require(best, "avg_power_watts"),
                        throughput=require(best, "output_throughput_tokens_per_sec"),
                        batch=batch,
                        is_moe="A3B" in model or "A22B" in model or "A35B" in model,
                    ))
                    print(f"{model:<35} {active_params:<8.1f} {best['avg_power_watts']:<8.0f} "
                          f"{batch:<8}")

        # Check hypothesis: models with sufficient active params are close to TDP
        results = all_results[gpu_type]
        if results:
            high_active = [r for r in results if r.active_params >= 8]
            if high_active:
                avg_power = sum(r.gpu_power for r in high_active) / len(high_active)
                print(f"\nModels with >=8B active params: avg power = {avg_power:.0f}W ({100*avg_power/tdp:.0f}% of TDP)")

    def draw():
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), tight_layout=True)

        for idx, (gpu_type, ax) in enumerate(zip(["B200", "H100"], axes)):
            results = all_results[gpu_type]
            tdp = 1000 if gpu_type == "B200" else 700

            ax.set_axisbelow(True)
            ax.grid(True, alpha=0.3)

            # Add TDP line
            ax.axhline(y=tdp, color="gray", linestyle="--", alpha=0.7, label=f"TDP ({tdp}W)")

            for r in results:
                color = COLORS["red"] if r.is_moe else COLORS["blue"]
                ax.scatter(r.active_params, r.gpu_power, s=150, c=color, edgecolors="black", zorder=3)

            ax.set_xlabel("Active Parameters (B)", fontsize=13)
            ax.set_ylabel("GPU Power (W)", fontsize=13)
            ax.set_title(f"{gpu_type}", fontsize=14)
            ax.tick_params(axis='both', labelsize=11)
            ax.set_xlim(0, None)
            ax.set_ylim(0, tdp * 1.15)

            legend_elements = [
                Patch(facecolor=COLORS["blue"], label="Dense"),
                Patch(facecolor=COLORS["red"], label="MoE"),
                Line2D([0], [0], color="gray", linestyle="--", label=f"TDP ({tdp}W)"),
            ]
            ax.legend(handles=legend_elements, fontsize=10, loc="lower right")

        return fig

    savefig(draw, "section3-1-model-size-power")

    return all_results["B200"]


# =============================================================================
# Section 3.2: Batch Size Affects Power
# =============================================================================

def section3_2_batch_size_power() -> dict[str, list[BatchPowerResult]]:
    """
    Analyze how batch size affects GPU power consumption and throughput per watt.

    Key finding: Increasing batch size increases GPU power draw, but throughput
    increases faster, resulting in better throughput per watt at higher batches.

    Note: Very low batch sizes (e.g., 8) sometimes show higher power than
    slightly larger batches (e.g., 16, 32). This may be due to GPU idle time
    between batches causing less efficient power states.
    """
    print_header("SECTION 3.2: Batch Size Affects Power")

    chat_configs = get_configs("lm-arena-chat")

    # Analyze multiple models to see the low-batch-size power anomaly
    target_models = [
        ("Qwen 3 8B", "B200", 1),
        ("Qwen 3 14B", "B200", 1),
        ("Qwen 3 32B", "B200", 1),
        ("Llama 3.1 8B", "B200", 1),
        ("Gemma 3 12B", "B200", 1),
    ]

    all_results: dict[str, list[BatchPowerResult]] = {}

    for model, gpu, num_gpus in target_models:
        configs = sorted(
            [c for c in filter_configs(chat_configs, nickname=model, gpu_model=gpu, num_gpus=num_gpus)
             if get_batch_size(c) >= MIN_BATCH_SIZE],
            key=lambda x: get_batch_size(x),
        )

        if not configs:
            continue

        print(f"\n{model} on {gpu} x{num_gpus}:")
        print(f"{'Batch':<10} {'Throughput':<12} {'GPU Power':<12} {'tok/s/W':<12}")
        print_separator(50)

        results: list[BatchPowerResult] = []
        for c in configs:
            batch = get_batch_size(c)
            tput = c["output_throughput_tokens_per_sec"]
            power = c["avg_power_watts"]
            tpw = tput / power
            results.append(BatchPowerResult(
                batch=batch,
                throughput=tput,
                gpu_power=power,
                tpw=tpw,
            ))
            print(f"{batch:<10} {tput:<12.0f} {power:<11.0f}W {tpw:<12.2f}")

        all_results[model] = results

    # Select models for plotting
    plot_models = ["Qwen 3 8B", "Qwen 3 32B", "Llama 3.1 8B", "Gemma 3 12B"]
    plot_models = [m for m in plot_models if m in all_results]

    colors_map = {
        "Qwen 3 8B": COLORS["blue"],
        "Qwen 3 32B": COLORS["red"],
        "Llama 3.1 8B": COLORS["green"],
        "Gemma 3 12B": COLORS["orange"],
    }
    markers_map = {
        "Qwen 3 8B": "o",
        "Qwen 3 32B": "s",
        "Llama 3.1 8B": "^",
        "Gemma 3 12B": "D",
    }

    def draw():
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), tight_layout=True)
        axes[0].set_axisbelow(True)
        axes[1].set_axisbelow(True)

        # Panel 1: Power vs Batch
        ax1 = axes[0]
        ax1.grid(True, alpha=0.3)
        for model in plot_models:
            results = all_results[model]
            batches = [r.batch for r in results]
            powers = [r.gpu_power for r in results]
            ax1.plot(batches, powers, f"{markers_map[model]}-", color=colors_map[model],
                    linewidth=2, markersize=8, label=model)

        ax1.set_xlabel("Batch Size", fontsize=13)
        ax1.set_ylabel("GPU Power (W)", fontsize=13)
        ax1.set_title("Batch Size vs GPU Power", fontsize=14)
        ax1.set_xscale("log", base=2)
        ax1.tick_params(axis='both', labelsize=11)
        ax1.legend(fontsize=10)

        # Panel 2: Throughput per Watt vs Batch
        ax2 = axes[1]
        ax2.grid(True, alpha=0.3)
        for model in plot_models:
            results = all_results[model]
            batches = [r.batch for r in results]
            tpws = [r.tpw for r in results]
            ax2.plot(batches, tpws, f"{markers_map[model]}-", color=colors_map[model],
                    linewidth=2, markersize=8, label=model)

        ax2.set_xlabel("Batch Size", fontsize=13)
        ax2.set_ylabel("Throughput per Watt (tok/s/W)", fontsize=13)
        ax2.set_title("Batch Size vs Throughput per Watt", fontsize=14)
        ax2.set_xscale("log", base=2)
        ax2.tick_params(axis='both', labelsize=11)
        ax2.legend(fontsize=10)
        ax2.set_ylim(bottom=0)

        return fig

    savefig(draw, "section3-2-batch-size-power")

    return all_results


# =============================================================================
# Section 3.3: Throughput per Watt Determines Cluster Capacity
# =============================================================================

def section3_3_throughput_per_watt() -> Section3_3Result:
    """
    Analyze throughput per watt for cluster capacity planning.

    Key finding: Throughput per watt (tok/s/W or req/s/W) determines how much
    service capacity you can pack into a fixed power budget. 17x difference
    between model classes.

    Uses a 2x scaling factor: cluster_power = gpu_power × 2 to account for
    CPU, DRAM, networking, cooling, and other infrastructure overhead.
    """
    print_header("SECTION 3.3: Throughput Per Watt Determines Cluster Capacity")

    print(f"\nNote: Using {CLUSTER_OVERHEAD}x overhead factor (cluster power = GPU power × {CLUSTER_OVERHEAD})")

    chat_configs = get_configs("lm-arena-chat")

    results: list[ThroughputResult] = []
    for c in chat_configs:
        tput_tokens = require(c, "output_throughput_tokens_per_sec")
        gpu_power = require(c, "avg_power_watts")
        avg_output_len = require(c, "avg_output_len")

        batch = get_batch_size(c)
        if tput_tokens > 0 and gpu_power > 0 and avg_output_len > 0 and batch >= MIN_BATCH_SIZE:
            cluster_power = gpu_power * CLUSTER_OVERHEAD
            tput_requests = tput_tokens / avg_output_len

            results.append(ThroughputResult(
                model=require(c, "nickname"),
                gpu=require(c, "gpu_model"),
                num_gpus=require(c, "num_gpus"),
                batch=batch,
                throughput_tokens=tput_tokens,
                throughput_requests=tput_requests,
                avg_output_len=avg_output_len,
                gpu_power=gpu_power,
                cluster_power=cluster_power,
                tpw_tokens=tput_tokens / cluster_power,
                tpw_requests=tput_requests / cluster_power,
            ))

    results.sort(key=lambda x: x.tpw_tokens, reverse=True)

    print("\nTop 10 by throughput per cluster watt (tok/s/W):")
    print_separator(100)
    print(f"{'Model':<30} {'GPU':<10} {'Batch':<8} {'tok/s/W':<12} {'req/s/W':<12}")
    print_separator(100)
    for r in results[:10]:
        print(f"{r.model[:28]:<30} {r.gpu}x{r.num_gpus:<6} {r.batch:<8} "
              f"{r.tpw_tokens:<12.3f} {r.tpw_requests:<12.3f}")

    # B200 vs H100 comparison
    print("\n--- B200 vs H100 Throughput Per Watt ---")

    h100_b200_pairs: list[H100B200Comparison] = []
    groups: dict[tuple, dict[str, ThroughputResult]] = {}
    for r in results:
        key = (r.model, r.batch)
        if key not in groups:
            groups[key] = {}
        groups[key][r.gpu] = r

    for key, gpu_configs in groups.items():
        if "H100" in gpu_configs and "B200" in gpu_configs:
            h100 = gpu_configs["H100"]
            b200 = gpu_configs["B200"]
            gain = (b200.tpw_tokens - h100.tpw_tokens) / h100.tpw_tokens * 100
            h100_b200_pairs.append(H100B200Comparison(
                model=key[0],
                batch=key[1],
                h100_tpw_tokens=h100.tpw_tokens,
                b200_tpw_tokens=b200.tpw_tokens,
                h100_tpw_requests=h100.tpw_requests,
                b200_tpw_requests=b200.tpw_requests,
                gain=gain,
            ))

    print(f"\n{'Model':<30} {'Batch':<8} {'H100 tok/s/W':<14} {'B200 tok/s/W':<14} {'B200 Gain':<10}")
    print_separator(80)
    for p in sorted(h100_b200_pairs, key=lambda x: -x.gain)[:8]:
        print(f"{p.model[:28]:<30} {p.batch:<8} {p.h100_tpw_tokens:<14.2f} "
              f"{p.b200_tpw_tokens:<14.2f} +{p.gain:.0f}%")

    # 1 MW capacity planning
    print("\n--- 1 MW Data Center Capacity Planning ---")
    power_budget = 1_000_000

    model_classes: dict[str, list[ThroughputResult]] = {
        "8B models": [r for r in results if "8B" in r.model and "A" not in r.model.split()[-1]],
        "MoE 30B A3B": [r for r in results if "A3B" in r.model],
        "Dense 32B": [r for r in results if "32B" in r.model and "A" not in r.model.split()[-1]],
        "Dense 70B": [r for r in results if "70B" in r.model],
        "Reasoning": [r for r in results if "V3" in r.model or "R1" in r.model],
    }

    print(f"\n{'Model Class':<20} {'tok/s/W':>12} {'req/s/W':>12} {'1MW tok/s':>14} {'1MW req/s':>14}")
    print_separator(75)
    for cls, entries in model_classes.items():
        if entries:
            best = max(entries, key=lambda x: x.tpw_tokens)
            cap_tok = best.tpw_tokens * power_budget
            cap_req = best.tpw_requests * power_budget
            print(f"{cls:<20} {best.tpw_tokens:>12.3f} {best.tpw_requests:>12.3f} "
                  f"{cap_tok/1e6:>12.2f} M/s {cap_req:>12,.0f}/s")

    # Generate plots - select top models ensuring both B200 and H100 are represented
    seen_models: set[str] = set()
    top_b200: list[ThroughputResult] = []
    top_h100: list[ThroughputResult] = []

    for r in results:
        if r.model not in seen_models:
            if r.gpu == "B200" and len(top_b200) < 4:
                top_b200.append(r)
                seen_models.add(r.model)
            elif r.gpu == "H100" and len(top_h100) < 4:
                top_h100.append(r)
                seen_models.add(r.model)

    top_models = top_b200 + top_h100

    # Select top models by req/s/W
    results_by_req = sorted(results, key=lambda x: x.tpw_requests, reverse=True)
    seen_req: set[str] = set()
    top_req_b200: list[ThroughputResult] = []
    top_req_h100: list[ThroughputResult] = []
    for r in results_by_req:
        if r.model not in seen_req:
            if r.gpu == "B200" and len(top_req_b200) < 4:
                top_req_b200.append(r)
                seen_req.add(r.model)
            elif r.gpu == "H100" and len(top_req_h100) < 4:
                top_req_h100.append(r)
                seen_req.add(r.model)
    top_req_models = top_req_b200 + top_req_h100

    # Select unique models for B200 vs H100 comparison
    seen_pairs: set[str] = set()
    unique_pairs: list[H100B200Comparison] = []
    for p in sorted(h100_b200_pairs, key=lambda x: -x.gain):
        if p.model not in seen_pairs and len(unique_pairs) < 5:
            seen_pairs.add(p.model)
            unique_pairs.append(p)

    def draw():
        fig, axes = plt.subplots(2, 2, figsize=(12, 9), tight_layout=True)
        for ax_row in axes:
            for ax in ax_row:
                ax.set_axisbelow(True)

        legend_elements = [
            Patch(facecolor=COLORS["blue"], label="B200"),
            Patch(facecolor=COLORS["red"], label="H100"),
        ]

        # Plot 1: Top models by tok/s/W (include both B200 and H100)
        ax1 = axes[0, 0]
        models_plot = [r.model for r in top_models]
        tpw_values = [r.tpw_tokens for r in top_models]
        bar_colors = [COLORS["blue"] if r.gpu == "B200" else COLORS["red"] for r in top_models]

        ax1.grid(True, axis='x', alpha=0.3)
        ax1.barh(range(len(models_plot)), tpw_values, color=bar_colors)
        ax1.set_yticks(range(len(models_plot)))
        ax1.set_yticklabels(models_plot, fontsize=11)
        ax1.set_xlabel("Throughput per Watt (tok/s/W)", fontsize=13)
        ax1.set_title("Top Models by tok/s/W", fontsize=14)
        ax1.tick_params(axis='x', labelsize=11)
        ax1.invert_yaxis()
        ax1.legend(handles=legend_elements, loc="lower right", fontsize=10)

        # Plot 2: Top models by req/s/W
        ax2 = axes[0, 1]
        models_req = [r.model for r in top_req_models]
        rpw_values = [r.tpw_requests for r in top_req_models]
        bar_colors_req = [COLORS["blue"] if r.gpu == "B200" else COLORS["red"] for r in top_req_models]

        ax2.grid(True, axis='x', alpha=0.3)
        ax2.barh(range(len(models_req)), rpw_values, color=bar_colors_req)
        ax2.set_yticks(range(len(models_req)))
        ax2.set_yticklabels(models_req, fontsize=11)
        ax2.set_xlabel("Throughput per Watt (req/s/W)", fontsize=13)
        ax2.set_title("Top Models by req/s/W", fontsize=14)
        ax2.tick_params(axis='x', labelsize=11)
        ax2.invert_yaxis()
        ax2.legend(handles=legend_elements, loc="lower right", fontsize=10)

        # Plot 3: B200 vs H100 comparison
        ax3 = axes[1, 0]
        x = np.arange(len(unique_pairs))
        width = 0.35

        ax3.grid(True, axis='y', alpha=0.3)
        ax3.bar(x - width/2, [p.h100_tpw_tokens for p in unique_pairs], width, label="H100", color=COLORS["red"])
        ax3.bar(x + width/2, [p.b200_tpw_tokens for p in unique_pairs], width, label="B200", color=COLORS["blue"])
        ax3.set_ylabel("Throughput per Watt (tok/s/W)", fontsize=13)
        ax3.set_title("B200 vs H100: tok/s/W", fontsize=14)
        ax3.set_xticks(x)
        ax3.set_xticklabels([p.model for p in unique_pairs], rotation=30, ha="right", fontsize=11)
        ax3.tick_params(axis='y', labelsize=11)
        ax3.legend(fontsize=10)
        ax3.set_ylim(bottom=0)

        # Plot 4: 1 MW capacity by model class
        ax4 = axes[1, 1]
        class_names = []
        capacities = []
        for cls, entries in model_classes.items():
            if entries:
                best_entry = max(entries, key=lambda x: x.tpw_tokens)
                class_names.append(cls)
                capacities.append(best_entry.tpw_tokens * power_budget / 1e6)

        ax4.grid(True, axis='y', alpha=0.3)
        colors4 = plt.cm.viridis(np.linspace(0.2, 0.8, len(class_names)))
        bars4 = ax4.bar(range(len(class_names)), capacities, color=colors4)
        ax4.set_xticks(range(len(class_names)))
        ax4.set_xticklabels(class_names, rotation=45, ha="right", fontsize=11)
        ax4.set_ylabel("Cluster Capacity (M tok/s)", fontsize=13)
        ax4.set_title("1 MW Data Center Capacity", fontsize=14)
        ax4.tick_params(axis='y', labelsize=11)
        ax4.set_ylim(bottom=0)

        for bar, val in zip(bars4, capacities):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                     f"{val:.1f}M", ha="center", va="bottom", fontsize=10)

        return fig

    savefig(draw, "section3-3-throughput-per-watt")

    return Section3_3Result(
        top_models=results[:10],
        h100_b200_comparison=h100_b200_pairs,
        model_classes=model_classes,
    )


# =============================================================================
# Run All Analyses
# =============================================================================

SECTIONS = {
    "1.1": section1_1_llm_tokens,
    "1.2": section1_2_mllm,
    "1.3": section1_3_diffusion,
    "2.1": section2_1_batch_size,
    "2.2": section2_2_moe_energy,
    "2.3": section2_3_b200_vs_h100,
    "2.4": section2_4_precision,
    "2.5": section2_5_multi_gpu_scaling,
    "3.1": section3_1_model_size_power,
    "3.2": section3_2_batch_size_power,
    "3.3": section3_3_throughput_per_watt,
}


def run_all(sections: list[str] | None = None) -> dict:
    """Run analysis functions and generate plots.

    Args:
        sections: List of section IDs to run (e.g., ["1.1", "2.3"]).
                  If None, runs all sections.
    """
    if sections is None:
        sections = list(SECTIONS.keys())

    print("\n" + "=" * 80)
    if len(sections) == len(SECTIONS):
        print("ML.ENERGY LEADERBOARD V3.0 - COMPLETE ANALYSIS")
    else:
        print(f"ML.ENERGY LEADERBOARD V3.0 - SECTIONS: {', '.join(sections)}")
    print("=" * 80 + "\n")

    results = {}
    for section_id in sections:
        if section_id not in SECTIONS:
            print(f"Warning: Unknown section '{section_id}', skipping")
            continue
        key = f"section{section_id.replace('.', '_')}"
        results[key] = SECTIONS[section_id]()

    print("\n" + "=" * 80)
    print(f"ANALYSES COMPLETE - Plots saved as SVG in {FIGURES_DIR}/")
    print("=" * 80)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run ML.ENERGY Leaderboard v3.0 blog analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available sections: {', '.join(SECTIONS.keys())}"
    )
    parser.add_argument(
        "sections",
        nargs="*",
        help="Section IDs to run (e.g., 1.1 2.3). If omitted, runs all sections."
    )

    args = parser.parse_args()
    run_all(args.sections if args.sections else None)
