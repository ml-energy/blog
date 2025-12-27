---
date: 2025-12-19
authors:
  - jaywonchung
categories:
  - measurement
  - energy
links:
  - The ML.ENERGY Leaderboard: https://ml.energy/leaderboard
  - The ML.ENERGY Benchmark: https://github.com/ml-energy/benchmark
---

# Reading the ML.ENERGY Leaderboard v3.0

We benchmarked 46 models across 7 tasks, producing 1,858 configurations on NVIDIA H100 and B200 GPUs.
This post summarizes our notable findings on LLM and diffusion model energy consumption.

<!-- more -->

## Energy by Architecture

What determines the energy consumption of generating one response?
For LLMs, a response is a complete answer to a prompt with all output tokens included.
For diffusion models, a response is one generated image or video.

### LLM

For LLMs, energy per response is simply energy per token multiplied by the number of output tokens.
The number of output tokens varies significantly by the model and task type.

**Task type determines output length.**
Different tasks naturally produce different output lengths.
This is particularly pronounced between two LLM tasks in our benchmark: Problem Solving (reasoning on) and Text Conversation (reasoning off).

<figure markdown>
  ![Energy and output length distributions](assets/ml-energy-leaderboard-v3.0/section1-1-llm-tokens-light.svg#only-light)
  ![Energy and output length distributions](assets/ml-energy-leaderboard-v3.0/section1-1-llm-tokens-dark.svg#only-dark)
  <figcaption>Distribution of energy per token, output length, and energy per response across models. Each data point is the model running at its maximum batch size on B200 GPUs.</figcaption>
</figure>

The main driver of energy per token is model size, but Problem Solving has a longer tail because long output lengths stress memory capacity and prevent large batch sizes, which increases energy per token due to lower GPU utilization.
On top of this, Problem Solving on average generates 11x more output tokens than Text Conversation (mean 6,998 vs. 621).
Therefore, Problem Solving responses consume on average 25x more energy than Text Conversation (mean 4,759 J vs. 191 J).

**Case study on Qwen 3 32B on 1x B200.**
Qwen 3 32B supports both reasoning mode and non-reasoning mode, allowing direct comparison of energy consumption for the same model on different tasks.

| Metric | Problem Solving | Text Conversation | Ratio |
|--------|----------------:|------------------:|-------|
| Max batch size | 128 | 512 | 4x lower |
| Average output tokens | 7,035 | 627 | 11x more |
| Energy/token @ max batch size | 0.312 J | 0.151 J | 2.1x higher |
| Energy/token @ batch size 128 | 0.312 J | 0.210 J | 1.5x higher |
| Energy/response | 2,192 J | 95 J | 23x more |

Longer output sequences in Problem Solving increases KV cache size, preventing the server from running larger batch sizes.
Therefore, when we compare energy per token at each task's maximum batch size, Problem Solving is 2.1x higher.
Even at the same batch size 128, longer sequences consumes more energy per token due to higher memory footprint.
Finally, combining longer outputs and higher energy per token results in 23x more energy per response for Problem Solving.

!!! Takeaway
    Task type heavily influences energy consumption.
    Notably, Problem Solving uses on average 25x more energy per response than Text Conversation.
    This comes from 11x more output tokens combined with higher energy per token due to memory pressure from long sequences.


### Multimodal LLM

Multimodal LLMs (MLLMs) takes images and/or videos alongside text as input and generates text responses.

<figure markdown>
  ![MLLM energy by modality](assets/ml-energy-leaderboard-v3.0/section1-2-mllm-light.svg#only-light)
  ![MLLM energy by modality](assets/ml-energy-leaderboard-v3.0/section1-2-mllm-dark.svg#only-dark)
  <figcaption>Energy per token by input modality (left) and batch size vs energy per token (right) on B200. Minimum-energy configurations are used.</figcaption>
</figure>

**Multimodality can increase energy consumption.**
The implications of multimodal inputs are twofold:
(1) the input image or video needs to be processed first on the CPU-side, which can take non-negligible time, and
(2) the model needs to run its vision encoder to convert them into vision tokens, which can sometimes increase input length significantly.

For the same model family, processing images uses 1-2x more energy per token than text, while video can use almost 6x more than text.

**Case study on Qwen 3 VL 8B on 1x B200.**
We're comparing Qwen 3 8B on Text Conversation with Qwen 3 VL 8B on Image Chat and Video Chat.
The vision encoder, which is part of prefill, and the increased input length are extra compute that costs more energy.
Furthermore, image and video preprocessing (e.g., converting raw image/video into tiles of pixels) run on the CPU.
This can become a bottleneck and limit the number of requests (batch size) running in the GPU, which underutilizes the GPU and increases energy per token.
Since video inputs are more expensive to preprocess than images, the bottleneck is worse.
This is a case where GPU energy consumption is not just about the GPU; the entire system and where the bottlenecks are matter a lot.


!!! Takeaway
    Multimodal inputs cost 1.5-6.6x more energy per token than text.
    The vision preprocessing pipeline in the CPU can create a bottleneck that prevents high batch sizes, reducing GPU utilization and increasing energy consumption.


### Diffusion Models

Diffusion models can generate images and videos from user text prompts.
Diffusion is where model size is not the best predictor of energy consumption due to multiple *runtime* factors: number of inference (denoising) steps, output resolution, and number of frames (for video).

<figure markdown>
  ![Text-to-image energy](assets/ml-energy-leaderboard-v3.0/section1-3-diffusion-light.svg#only-light)
  ![Text-to-image energy](assets/ml-energy-leaderboard-v3.0/section1-3-diffusion-dark.svg#only-dark)
  <figcaption>Energy per image/video for diffusion models (minimum-energy configuration on B200).</figcaption>
</figure>

**Text-to-image varies 20x** across models ranging from 0.6B to 12B parameters and 20-50 denoising steps.
Notably, Hunyuan-DiT 1.2 (1.5B parameters) consumes more energy than SD 3.5 Large (8.1B parameters) despite having fewer parameters, due in large part to running 50 denoising steps versus 28.

**Text-to-video can be very energy intensive**, with output resolution and frame count among the dominant factors in energy consumption.
CogVideoX 1.5 5B uses more energy than Wan 2.1 14B despite being a smaller model, largely because it generates at higher resolution (768x1360 vs 480x832).
HunyuanVideo stands out at 1.16 MJ because it generates 720p video at 129 frames, which is significantly more total pixels than Wan 2.1 14B (which generates 480p at 81 frames), resulting in 4x higher energy despite similar model sizes (13B vs 14B).

!!! Takeaway
    Diffusion model energy depends on more than model size: number of denoising steps, output resolution, and frame count matter as much or more.
    Video generation can consume orders of magnitude more energy than image generation.


## Deeper Dive into Energy

Let's dive deeper into the factors that affect energy consumption.

### Batch Size

Increasing batch size reduces energy per token by 3-5x, but it's not free.

<figure markdown>
  ![Batch size effect](assets/ml-energy-leaderboard-v3.0/section2-1-batch-size-light.svg#only-light)
  ![Batch size effect](assets/ml-energy-leaderboard-v3.0/section2-1-batch-size-dark.svg#only-dark)
  <figcaption>Scaling trends of energy per token, token generation throughput, median ITL, and power draw with increasing batch size. Metrics normalized to % of maximum, except power which is normalized to % of GPU TDP (1000W per B200).</figcaption>
</figure>

GPUs achieve peak efficiency when fully utilized; small batches leave compute units idle and waste static energy consumption, and larger batches amortize fixed costs (e.g., memory transfers) across more work.
Therefore, as batch size increases, energy per token drops sharply at first, then plateaus.
Batch size is hard-capped by GPU memory capacity.

The energy efficiency gains of increasing batch size is not without tradeoffs.
Latency (median ITL in this analysis) increases with batch size, as there is strictly more work to do for each batch.[^latency-and-batch-size]
Throughput also increases with batch size, but with diminishing returns as GPU utilization reaches saturation.[^latency-and-throughput]
Finally, power draw increases with batch size, as the GPU's compute and memory capacity are more fully utilized and actively doing work.

[^latency-and-batch-size]: For very small batch sizes, LLM weight loading latency will dominate, so increasing batch size may not increase latency very much.
[^latency-and-throughput]: Depending on the model, the GPU's memory capacity may reach saturation before compute utilization, and throughput gains will not diminish as quickly.

!!! Takeaway
    Batch size is a critical lever for time, throughput, and energy.
    3-5x energy reduction is achievable by increasing batch size, with throughput gains showing diminishing returns.


### MoE Architecture

With the Mixture-of-Experts (MoE) architecture, the total number of parameters is less of a determinant of energy consumption; the number of *active* parameters is important.

<figure markdown>
  ![MoE energy efficiency](assets/ml-energy-leaderboard-v3.0/section2-1-moe-energy-light.svg#only-light)
  ![MoE energy efficiency](assets/ml-energy-leaderboard-v3.0/section2-1-moe-energy-dark.svg#only-dark)
  <figcaption>Energy per token by active parameters for Qwen 3 model variants on B200 (Problem Solving task, minimum-energy configuration).</figcaption>
</figure>

Within the Qwen 3 model family, we compare two MoE variants, 30B A3B (30B total, 3B active) and 235B A22B (235B total, 22B active), and three dense variants, 8B, 14B, and 32B.
For dense models, energy per token increases with model size (total number of parameters).
However, when we throw in MoE models, we see that their energy per token is much lower than what a dense model of similar total size would consume.
For instance, the energy per token of Qwen 3 30B A3B is 3.56x lower than that of Qwen 3 32B, despite having a similar total number of parameters.
Qwen 3 235B A22B consumes more energy than 32B as it needs to use more GPUs to fit total parameters in memory, but still far less than what a dense 235B model would.

!!! Takeaway
    MoE models consume much less energy compared to dense models of similar total number of parameters.
    The number of active parameters is an important factor, though other factors like memory pressure also play a role.


### B200 versus H100

How much is B200 better than H100 in terms of energy?
For each model, let's compare the minimum-energy configuration on B200 against H100 that meets latency constraints.

<figure markdown>
  ![B200 vs H100](assets/ml-energy-leaderboard-v3.0/section2-2-b200-vs-h100-light.svg#only-light)
  ![B200 vs H100](assets/ml-energy-leaderboard-v3.0/section2-2-b200-vs-h100-dark.svg#only-dark)
  <figcaption>B200 vs H100 energy comparison at matched latency constraints. Percentage of B200 energy reduction over H100 is annotated on each bar.</figcaption>
</figure>

We also compared the two GPUs for all models with three different latency constraints (50/100/250 ms median ITL for LLMs, 10/30/60 s for text-to-image, 100/500/1000 s for text-to-video).

**LLM.** Across all three median ITL constraints, B200 wins 88% (63/72) of comparisons with a median 35% energy reduction (-53% to +82%).
A few notable exceptions happen at tight latency constraints.
Namely, while B200's large VRAM allows fitting large models on fewer GPUs without inter-GPU communication overhead, at tight latency constraints, using more H100 GPUs with higher degree of parallelism can be more energy efficient.
For example, at the 50 ms constraint: Qwen 3 30B A3B Thinking uses 53% less energy on H100 (2 GPUs, batch size 128) vs B200 (1 GPU, batch size 64), and Qwen 3 235B A22B Instruct FP8 uses 33% less energy on H100 (8 GPUs, batch size 192) vs B200 (2 GPUs, batch size 64).
At relaxed constraints (60 ms or larger), B200 wins as communication overhead is smaller and higher batch sizes become feasible.

**Diffusion (Text to Image).** Across all three latency constraints, B200 wins 86% (18/21) of comparisons with a median 15% energy reduction (-4% to +23%).
Stable Diffusion 3.5 Medium is the outlier, but still, H100 wins only by a small margin (by 4%).

**Diffusion (Text to Video).** Across all three latency constraints, B200 wins 79% (11/14) of comparisons with a median 4% energy reduction (-6% to +8%).
In general, we can say that B200 and H100 consume similar energy for video generation.
B200 still provides lower latency, which matters for user experience.


!!! Takeaway
    B200 achieves lower energy in 79-88% of comparisons at matched latency constraints.
    Particularly for LLMs, B200 provides a median 35% energy reduction, but with tight latency constraints, using more H100 GPUs with higher parallelism can consume less energy.


### Precision

FP8 typically reduces both energy and latency at practical batch sizes, but loses at low batch sizes due to fixed overhead that isn't amortized.

<figure markdown>
  ![Precision comparison](assets/ml-energy-leaderboard-v3.0/section2-4-precision-light.svg#only-light)
  ![Precision comparison](assets/ml-energy-leaderboard-v3.0/section2-4-precision-dark.svg#only-dark)
  <figcaption>FP8 loses at batch size 8-16, then wins at batch sizes from 32. The dashed vertical lines mark the crossover point.</figcaption>
</figure>

When we compare models with official weights in both BF16 and FP8 on the same GPU model, number of GPUs, and batch size, we see the following:

- **Batch size 8-16:** FP8 wins 0% on energy, 0% on latency
- **Batch size 32-64:** FP8 wins 60% on energy, 80% on latency
- **Batch size 65-256:** FP8 wins 100% on energy, 83% on latency

At batch size 8-16, FP8 has higher latency (up to 22% slower) and higher energy (up to 61% higher).
FP8 kernels have extra computation and memory overhead (e.g., for dequantization/rescaling) that do not get amortized well at small batch sizes, leading to higher time and energy at low batch sizes.
As we grow the batch size, FP8's energy advantage comes more gradually at relatively larger batch sizes than latency.
We believe this is because GPUs are capable of doing more FP8 compute throughput than BF16, and at the same batch size, FP8 underutilizes the GPU more, leading to higher energy consumption until batch size is large enough to saturate the GPU.

There is a notable anomaly in our benchmark: Qwen 3 Coder 480B A35B (Code Completion) on 8x B200.
FP8 consistently consumes more time and more energy than BF16 across all batch sizes.
This is because due to a limitation in vLLM ([vLLM Recipes](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-Coder-480B-A35B.html#fp8-models); last accessed 2024-12-26), the FP8 model was required to run attention with data parallelism, while BF16 could run attention with tensor parallelism.
Attention data parallelism, especially without Prefill-Decode disaggregation, incurs load imbalance between GPUs that are assigned very different sequence lengths (e.g., some running long prefills whereas others run decode).
Since the straggler GPU bottlenecks the entire batch, this can lead to significant latency overhead.
Furthermore, the non-straggler GPUs do nothing and waste static power waiting for the straggler, leading to even higher energy consumption as well.

!!! Takeaway
    FP8 typically wins at batch sizes north of 32 on both energy and latency.
    At batch size 8-16, FP8's fixed overhead makes it slower and less efficient than BF16.


### Multi-GPU Scaling

We can execute a model on different numbers of GPUs, which affects both latency and energy consumption.
GPT OSS 120B is a good case study.

<figure markdown>
  ![Multi-GPU scaling](assets/ml-energy-leaderboard-v3.0/section2-5-multi-gpu-light.svg#only-light)
  ![Multi-GPU scaling](assets/ml-energy-leaderboard-v3.0/section2-5-multi-gpu-dark.svg#only-dark)
  <figcaption>Time-energy tradeoff for GPT OSS 120B on B200 (left) and H100 (right). In both cases, scaling from 1 GPU (blue line) to 2 GPUs (red line) at fixed batch size trades energy for time. On H100 particularly, 1 GPU is memory-limited to batch size 64, while 2 GPUs unlock batch size 2048, which achieves much lower energy.</figcaption>
</figure>

**At the same batch size, more GPUs trade energy for latency.**
In general, increasing parallelism with more GPUs reduces latency but also increases energy at the same batch size because (1) latency does not decrease linearly due to communication overhead, and (2) less compute per GPU can lead to lower GPU utilization.
On B200 configurations, adding GPUs at the same batch size reduced latency in 81% of cases and *always* increased energy per token.
Similarly, on H100 configurations, energy increases in 93% of the cases and latency *always* decreases.

**Energy savings require models that need the extra memory.**
On top of the above, in cases where adding more GPUs *enables* larger batch sizes due to increased aggregate memory capacity, we can see energy reductions.
For GPT OSS 120B on 1x B200 with a 180 GB VRAM, the model already fits at high batch sizes on 1 GPU (batch size 3072), so 2 GPUs only add overhead without enabling lower energy.
On 1x H100 with an 80 GB VRAM, the server is limited to batch size 64, while 2 GPUs unlock batch size 2048 and achieve 68% lower minimum energy.
So, the model's total parameter memory footprint relative to the GPU's memory capacity is an important factor in determining whether multi-GPU scaling can reduce energy by enabling larger batch sizes.

!!! Takeaway
    At the same batch size, more GPUs typically reduce latency but increase energy).
    When adding GPUs enables larger batch sizes, energy can be reduced--but only if serving was previously limited by memory capacity.


## What about Power?

How does power consumption vary, and what does it mean for data center capacity?

### Model Size and Power

Generally, larger models draw more power, and the number of *active parameters* is an important factor (Mixture-of-Experts (MoE) have less active parameters than total parameters).
When the GPU is fully utilized with sufficient amount of compute, average GPU power saturates near the GPU's Thermal Design Power (TDP).

<figure markdown>
  ![Model size and power](assets/ml-energy-leaderboard-v3.0/section3-1-model-size-power-light.svg#only-light)
  ![Model size and power](assets/ml-energy-leaderboard-v3.0/section3-1-model-size-power-dark.svg#only-dark)
  <figcaption>GPU power draw against the number of active parameters for various Text Conversation models on 1x B200 (left) and 1x H100 (right). For each model, the configuration with the largest possible batch size on that GPU is shown.</figcaption>
</figure>

It is worth noting that when a model is large and the deployment configuration is memory, it cannot reach high batch sizes, leaving compute underutilized and power below TDP even for larger models.
Specifically, on H100, Gemma 3 27B and Qwen 3 32B are both limited to batch size 32, drawing only 82% and 86% of TDP respectively.
In contrast, on B200 with more memory headroom, the same models reach batch sizes 256-512 and their power draw is close to the GPU's TDP.

!!! Takeaway
    Models with more active parameters draw more power, up to the GPU's TDP.


### Batch Size and Power

Higher batch sizes increase GPU power, but throughput increases faster, so throughput per watt improves.

*Qwen 3 8B on B200 x1* (Text Conversation, batch sizes >=16):

| Batch | Throughput | GPU Power | Throughput/Watt |
|-------|------------|-----------|-----------------|
| 32 | 2,610 tok/s | 494 W | 5.28 |
| 64 | 4,116 tok/s | 543 W | 7.58 |
| 256 | 9,570 tok/s | 831 W | 11.52 |
| 1536 | 12,393 tok/s | 801 W | 15.47 |

From batch size 32 to 1536: power increases 1.6x, throughput increases 4.7x, throughput per watt improves 2.9x.

<figure markdown>
  ![Batch size and power](assets/ml-energy-leaderboard-v3.0/section3-2-batch-size-power-light.svg#only-light)
  ![Batch size and power](assets/ml-energy-leaderboard-v3.0/section3-2-batch-size-power-dark.svg#only-dark)
  <figcaption>GPU power and throughput per watt vs batch size.</figcaption>
</figure>

!!! Takeaway
    Higher batch sizes increase GPU power, but throughput grows faster.
    Result: 2-3x improvement in throughput per watt at higher batch sizes.


### Throughput per Watt

For power-constrained data centers, model choice affects capacity by 17-37x.

Text Conversation task.
Cluster power is GPU power x 2 (accounting for CPU, DRAM, networking, and cooling).
We selected the maximum tok/s/W configuration per model:

| Model | GPU | Batch | tok/s/W | req/s/W |
|-------|-----|-------|---------|---------|
| Llama 3.1 8B | B200 | 2048 | 7.83 | 0.020 |
| Qwen 3 8B | B200 | 1536 | 7.74 | 0.012 |
| Qwen 3 30B A3B (MoE) | B200 | 1024 | 6.52 | 0.007 |
| Qwen 3 14B | B200 | 1024 | 5.56 | 0.009 |
| Qwen 3 32B | H100 | 256 | 1.58 | 0.003 |

**B200 delivers 59-117% better throughput per watt than H100:**

| Model | Batch | H100 | B200 | Gain |
|-------|-------|------|------|------|
| Qwen 3 235B A22B FP8 | 256 | 0.37 tok/s/W | 0.80 tok/s/W | +117% |
| Qwen 3 30B A3B | 1024 | 3.48 tok/s/W | 6.52 tok/s/W | +87% |
| Qwen 3 32B | 256 | 1.58 tok/s/W | 2.88 tok/s/W | +82% |

**1 MW data center capacity:**

| Model Class | tok/s/W | req/s/W | 1 MW tok/s | 1 MW req/s |
|-------------|---------|---------|------------|------------|
| 8B models | 7.83 | 0.020 | 7.83 M/s | 19,800/s |
| MoE 30B A3B | 6.52 | 0.007 | 6.52 M/s | 7,000/s |
| Dense 32B | 2.88 | 0.005 | 2.88 M/s | 4,600/s |
| Dense 70B | 1.42 | 0.004 | 1.42 M/s | 4,000/s |
| Reasoning | 0.45 | 0.001 | 0.45 M/s | 535/s |

8B models deliver 17x more tok/s and 37x more req/s per watt than reasoning configurations.

<figure markdown>
  ![Throughput per watt](assets/ml-energy-leaderboard-v3.0/section3-3-throughput-per-watt-light.svg#only-light)
  ![Throughput per watt](assets/ml-energy-leaderboard-v3.0/section3-3-throughput-per-watt-dark.svg#only-dark)
  <figcaption>Throughput per watt across models, GPUs, and model classes.</figcaption>
</figure>

!!! Takeaway
    For operators: model choice dominates capacity planning.
    A 1 MW cluster serves 19,800 req/s with 8B models versus 535 req/s with reasoning enabled--a 37x difference.


## Summary

**For ML researchers:**

- Output length is the primary energy driver: same model, 40x energy difference between tasks
- Energy scales with active parameters, not total: a 30B MoE with 3B active uses 2x less energy than an 8B dense model
- FP8 is not always better: attention DP overhead can make BF16 2-6x more efficient

**For ML systems engineers:**

- B200 delivers 16-78% lower energy per token over H100 (up to 117% better throughput per watt)
- Batch size is your biggest lever: 3-5x energy per token reduction
- Multi-GPU is a tradeoff: more GPUs can reduce energy but often increases latency 19-255%
- MXFP4 on B200 achieves the lowest energy per token (0.028 J) in the benchmark

**For operators:**

- Match model to task: do not enable reasoning mode for simple tasks
- Consider total cost: energy per token x tokens per response x queries per day
- New hardware pays off: B200 investment recovers through 59-117% throughput per watt gains
- Two capacity metrics matter: tok/s/W for raw throughput, req/s/W for user-facing capacity


*[batch size]: The number of requests running concurrently. For LLM and MLLMs with variable number of requests running over time, batch size means the maximum number of requests (`max_num_seqs`) configured for the server. For diffusion models, batch size is exactly the number of items (e.g., images, videos) being generated together.
