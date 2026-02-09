---
date: 2026-02-09
authors:
  - jaywonchung
categories:
  - measurement
  - energy
links:
  - The ML.ENERGY Leaderboard: https://ml.energy/leaderboard
  - The ML.ENERGY Benchmark: https://github.com/ml-energy/benchmark
---

# LLM Inference Energy: A Longitudinal Analysis

The ML.ENERGY Leaderboard went from v2.0 (September 2024) to [v3.0](https://ml.energy/leaderboard) (December 2025) with major changes: up-to-date models, hardware, software, and datasets.
The [v3.0 blog post](ml-energy-leaderboard-v3.0.md) covered the details of the v3.0 results, but how to they compare to the times v2.0?
Are we making progress on energy efficiency?
In this short post, we would like to look at the impact of **software optimizations** on energy efficiency over time, using the Llama 3.1 family as a case study.

<!-- more -->

## The Case of Llama 3.1

### Energy Reduction

To reason about how the software stack affects energy, we fix the model and hardware: the Llama 3.1 family (8B, 70B, 405B) on H100 GPUs with the same number of GPUs.

<figure markdown>
  ![Energy comparison](assets/llm-energy-over-time/llama-energy-per-token-light.svg#only-light)
  ![Energy comparison](assets/llm-energy-over-time/llama-energy-per-token-dark.svg#only-dark)
  <figcaption>Energy per output token vs. batch size for Llama 3.1 models on H100 GPUs. For 405B, V2 uses BF16 and V3 uses FP8.</figcaption>
</figure>

The 8B and 70B models are exactly the same in V2 and V3.
For the 405B model, we added one more layer: V3 uses native FP8 quantization.

At small batch sizes, V3 uses substantially less energy per token than V2.
For instance, the 8B model on batch size 64 reduces energy per token by 41% (0.20 J to 0.12 J).
The 70B model shows consistent improvements across all batch sizes, with up to a 15% reduction.
Similarly, the 405B model (especially with FP8) shows up to a 39% energy reduction at a batch size of 256.
V2 or V3, the mathematical operations carried out by a small batch size are the same.
Energy reductions come from software optimizations in vLLM (0.5.4 in V2 vs. 0.11.1 in V3).
Notably, at small batch sizes where the latency of one iteration is low, [CPU-side overheads](ml-energy-leaderboard-v3.0.md#multimodal-llm) may bottleneck execution and underutilize the GPU, increasing [static power wastage](ml-energy-leaderboard-v3.0.md#static-power-wastage) and energy consumption.
Software optimizations like vLLM Asynchronous Scheduling can mitigate this significantly.

At larger batch sizes, the energy gap narrows.
This is likely as batch size increases, the GPU becomes more fully utilized (with the same computations), and constant software-level loverheads are less significant relative to GPU computations.

To understand things deeper, let's break down energy per token into its components:

$$
\frac{\text{Energy}}{\text{Token}}
= \frac{\text{Power} \cdot \text{Time}}{\text{Token}}
= \frac{\text{Power}}{\text{Token Throughput}}
$$

Let's look at each of these components.


### Token Throughput

<figure markdown>
  ![Throughput comparison](assets/llm-energy-over-time/llama-throughput-light.svg#only-light)
  ![Throughput comparison](assets/llm-energy-over-time/llama-throughput-dark.svg#only-dark)
  <figcaption>Token throughput vs. batch size for Llama 3.1 models on H100 GPUs.</figcaption>
</figure>

At matched batch sizes, **V3 achieves 3-5x higher throughput** across all three model sizes.
Progress in software optimizations (especially vLLM) over more than a year is the main driver of this improvement.
V2's throughput curve is notably flat across batch sizes, suggesting the older vLLM version was unable to fully exploit increased parallelism from larger batches.
V3 shows the expected pattern: throughput climbs steeply with batch size and eventually saturates as GPU compute and memory bandwidth become the bottleneck.

With 3-5x throughput gains, one might expect proportional energy reductions.
But as we saw, energy per token improved only 15-41%.
Something is offsetting the throughput gains.

### Power

<figure markdown>
  ![Power comparison](assets/llm-energy-over-time/llama-power-light.svg#only-light)
  ![Power comparison](assets/llm-energy-over-time/llama-power-dark.svg#only-dark)
  <figcaption>Average GPU power vs. batch size for Llama 3.1 models on H100 GPUs.</figcaption>
</figure>

vLLM GPU power draw is substantially higher in V3 than in V2.
Power draw is a good indicator of the GPU's utilization as it reflect hardware circuit activity directly, which is the core factor that drove throughput improvements.[^v2-power]

[^v2-power]: Full disclosure: V2 only measured power during the entire benchmark duration (not explicitly during the steady-state window), which underestimates V2's power draw. This gets worse at larger batch sizes, where the relative duration of the steady state takes up a shorter portion of the entire duration. However, even with some underestimation of V2's power, the gap is large enough to conclude that vLLM draws more power in V3 than in V2.

### Putting It Together

V3's vLLM improvements keep the GPU busier, which both increases throughput (more tokens per second) and increases power draw (more of the GPU's circuitry is active).
Since energy per token is the ratio of these two quantities, the energy improvement depends on which one grows faster.
Throughput grew faster than power, resulting in a net energy reduction.

!!! Takeaway
    V3 achieves 3-5x higher throughput and 15-41% lower energy per token on Llama 3.1 models with the same hardware.
    Especially, at small batch sizes, we see nearly a halving of energy per token, which is a significant improvement for latency-sensitive applications.
    The reason energy per token improves less than throughput is that power also increases as the GPU is more fully utilized.


## Summing Up

Energy consumption is a pressing issue across the AI stack, from individual inference requests to datacenter-scale deployments.
But the V2-to-V3 transition shows that we are making relentless progress: software optimizations (vLLM 0.5.4 to 0.11.1) delivered 3-5x throughput improvements and measurable energy reductions.

The small batch size (i.e., low latency) regime is the most challenging to run efficiently, and as we saw, that is also where software optimizations had the biggest impact on energy efficiency.
This regime actually matters a lot for *agentic* applications that generate a *ton* of tokens; in order to keep latency in a reasonable range, these applications often run with small batch sizes.
In this specific analysis, we see nearly a halving of energy per token at smaller batch sizes, which is a significant improvement for these applications.

Progress spans the entire stack (inference engines, model architectures, quantization, and hardware) and the improvements compound.
To understand how all these factors interact and to learn how to reason about and explain energy consumption, see [our V3 leaderboard blog post](ml-energy-leaderboard-v3.0.md).


*[batch size]: The number of requests running concurrently. For LLM and MLLMs with variable number of requests running over time, batch size means the maximum number of requests (`max_num_seqs`) configured for the server. For diffusion models, batch size is exactly the number of items (e.g., images, videos) being generated together.
*[batch sizes]: The number of requests running concurrently. For LLM and MLLMs with variable number of requests running over time, batch size means the maximum number of requests (`max_num_seqs`) configured for the server. For diffusion models, batch size is exactly the number of items (e.g., images, videos) being generated together.
