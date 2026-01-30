---
date: 2026-01-13
authors:
  - jaywonchung
categories:
  - measurement
  - energy
links:
  - The ML.ENERGY Leaderboard: https://ml.energy/leaderboard
  - The ML.ENERGY Benchmark: https://github.com/ml-energy/benchmark
  - PDF Version: https://arxiv.org/abs/2601.22076
---

# Reading the ML.ENERGY Leaderboard v3.0

With [The ML.ENERGY Benchmark v3.0](https://github.com/ml-energy/benchmark/releases/tag/v3.0) we released in December 2025, we expanded our scope to up-to-date important models, tasks, and GPU hardware.
This included 46 models across 7 tasks, producing 1,858 configurations on NVIDIA H100 and B200 GPUs.[^software-setup]
As always, latest benchmarking results are public and can be browsed on [The ML.ENERGY Leaderboard](https://ml.energy/leaderboard).

In this post, we first present empirical observations from measurements, and then develop a reasoning framework that explains *why* we observe certain energy behaviors.

<!-- more -->

A PDF version of this post is available on [arXiv](https://arxiv.org/abs/2601.22076).
For more details on our benchmarking methodology, please refer to [our NeurIPS 25 D&B paper](https://arxiv.org/abs/2505.06371).

[^software-setup]: We used vLLM 0.11.1 for LLM/MLLMs and xDiT 0.4.5 for diffusion models.

## Energy by Architecture

What determines the energy consumption of generating one **response**?
For Large Language Models (LLMs), a response is a complete answer to a prompt with all output tokens included.
For diffusion models, a response is one generated image or video.

### LLM

**Task type heavily influences output length.**
LLM time and energy consumption is dominated by the decoding (token generation) phase.
Different tasks naturally produce different distributions of output lengths.
This is particularly pronounced between two LLM tasks in our benchmark: Problem Solving (reasoning on) and Text Conversation (reasoning off).

<figure markdown>
  ![Energy and output length distributions](assets/ml-energy-leaderboard-v3.0/section1-1-llm-light.svg#only-light)
  ![Energy and output length distributions](assets/ml-energy-leaderboard-v3.0/section1-1-llm-dark.svg#only-dark)
  <figcaption>Distribution of (a) number of output tokens, (b) energy per token, and (c) energy per response across all models on B200 GPUs using their respective minimum-energy configurations.</figcaption>
</figure>

Here, we're comparing the minimum-energy configuration[^minimum-energy-config] of each model on B200 GPUs, which allows us to focus on model and task differences without being confounded by hardware utilization differences.
Problem Solving generates on average 10x more output tokens than Text Conversation (mean 6,988 vs. 717).
Additionally, longer output sequences stress memory capacity and prevent larger batch sizes, increasing energy per token due to lower GPU utilization.
Since energy per response is energy per token multiplied by the number of output tokens, these two factors multiply, resulting in Problem Solving consuming on average 25x more energy per response than Text Conversation (mean 4,625 J vs. 184 J).

[^minimum-energy-config]: The configuration (e.g., batch size, number of GPUs) that achieves the lowest energy consumption for the model.

**Case study on Qwen 3 32B on 1x B200.**
Qwen 3 32B supports both reasoning and non-reasoning, enabling direct comparison on the same model.

| Metric | Text Conversation | Problem Solving | Ratio |
|--------|------------------:|----------------:|------:|
| Max batch size (BS) | 512 | 128 | 0.25x |
| Mean output tokens | 627 | 7,035 | 11x |
| Energy/token @ BS 128 | 0.209 J | 0.312 J | 1.5x |
| Energy/token @ max BS | 0.151 J | 0.312 J | 2.1x |
| Energy/response | 95 J | 2,192 J | 23x |

Longer output sequences in Problem Solving increase the amount of KV cache memory usage per response, preventing larger batch sizes.
Therefore, when we compare energy per token at each task's maximum batch size, Problem Solving is 2.1x higher.
Even at the same batch size (128), longer sequences consume more energy per token due to higher memory footprint.
Finally, combining longer outputs and higher energy per token results in 23x energy per response for Problem Solving for this model.

!!! Takeaway
    Task type heavily influences energy consumption.
    Notably, Problem Solving (reasoning) uses on average 25x more energy per response than Text Conversation.
    This comes from 10x more output tokens combined with higher energy per token due to memory pressure limiting batch size.


### Multimodal LLM

Multimodal LLMs (MLLMs) take images and/or videos alongside text as input and generate text responses.

<figure markdown>
  ![MLLM energy by modality](assets/ml-energy-leaderboard-v3.0/section1-2-mllm-light.svg#only-light)
  ![MLLM energy by modality](assets/ml-energy-leaderboard-v3.0/section1-2-mllm-dark.svg#only-dark)
  <figcaption>(a) shows three models from the Qwen 3 family across three modalities (minimum-energy configurations), and (b) shows how modality affects batch size and KV cache utilization for the 8B model, showing why energy per token increases. Text modality uses the text-only model (e.g., Qwen 3 8B), whereas Image and Video use the vision-language variant (e.g., Qwen 3 VL 8B).</figcaption>
</figure>

**Multimodality can increase energy.**
The implications of multimodal inputs are threefold:

1. Models run their modality encoder to convert inputs into multimodal tokens, which increases computation and memory operations and therefore energy consumption.
2. In GPU memory-constrained scenarios, the modality encoder and the increase in input length increase memory usage, which can limit batch size.
3. Multimodal inputs need to be preprocessed first on the CPU-side (e.g., converting raw image/video into tiles of pixels), which can take non-negligible time and become a bottleneck that further limits batch size.

Indeed, when we compare minimum-energy configurations for different modalities, text + image inputs use 1.1-5.2x the energy per token of text, while text + video inputs use 1.3-15.0x.[^mllm-task-output-length]

**Case study on Qwen 3 (VL) 8B on 1x B200.**
We compare Qwen 3 8B on Text Conversation with Qwen 3 VL 8B on Image Chat and Video Chat tasks.
For this smaller 8B model, the overheads of vision encoders and CPU-side preprocessing limit batch size significantly and underutilize the GPU.
In particular, video inputs typically get converted to more vision tokens and are more expensive to preprocess on the CPU side, as shown by the much smaller batch size and higher energy per token.
The drop in KV cache utilization as vision preprocessing overhead grows confirms that GPU memory was not the limiting factor---there was spare capacity for more tokens---but CPU-side vision preprocessing became a severe bottleneck that limited batch size.

All in all, this is a case where GPU energy consumption is not just about the GPU; the entire system and the location of bottlenecks matter.[^gpu-multimodal-processing]
If CPU-side processing speed remains unchanged and only the GPU is upgraded, the GPU will only be more underutilized.
In subsequent analyses, we do not include MLLMs because CPU-side bottlenecks make it difficult to isolate factors that impact GPU energy.

[^mllm-task-output-length]: One caveat of this cross-modality comparison is that, as we have seen in the LLM section, different tasks can have different output lengths that affect energy per token. For the models we compared, Text Conversation, Image Chat, and Video Chat have average output lengths of 808, 944, and 392 tokens, respectively. This isn't as large as the difference between Text Conversation and Problem Solving and shouldn't affect energy per token as much as that case, but Video Chat's shorter output length (which allows requests to finish faster and reduces batch size when the CPU is the bottleneck) may have increased energy per token compared to Image Chat and Text Conversation.
[^gpu-multimodal-processing]: There are some proposals to run vision preprocessing on the GPU itself (e.g., [vLLM #21995](https://github.com/vllm-project/vllm/issues/21995)), which can alleviate CPU-side bottlenecks but instead shift compute more to the GPU, which will likely introduce its own interesting tradeoffs.

!!! Takeaway
    Multimodal inputs cost 1.1-5.2x (image) and 1.3-15.0x (video) the energy per token of text.
    CPU-side vision preprocessing can be a bottleneck that reduces batch size and increases energy per token.


### Diffusion Models

We benchmarked diffusion models that generate images and videos from user text prompts.
Diffusion is where model size is not the best predictor of energy consumption due to multiple *runtime* factors: number of inference (denoising) steps, output resolution, and number of frames (for video).

<figure markdown>
  ![Text-to-image energy](assets/ml-energy-leaderboard-v3.0/section1-3-diffusion-light.svg#only-light)
  ![Text-to-image energy](assets/ml-energy-leaderboard-v3.0/section1-3-diffusion-dark.svg#only-dark)
  <figcaption>Energy per image/video for diffusion models (minimum-energy configuration on B200). SD is short for Stable Diffusion.</figcaption>
</figure>

**Text-to-image varies 20x across models.**
Models range from 0.6B to 12B parameters with 20-50 denoising steps.
Notably, Hunyuan-DiT 1.2 (1.5B) consumes more energy than SD 3.5 Large (8.1B) despite fewer parameters, largely due to running 50 vs. 28 denoising steps.

**Text-to-video can be very energy intensive.**
Generating a single video consumes 26 kJ to 1.16 MJ---one to two orders of magnitude more than images.
CogVideoX 1.5 5B uses more energy than Wan 2.1 14B despite being smaller, largely because it generates at higher resolution (768x1360 vs. 480x832).
HunyuanVideo reaches 1.16 MJ because it generates 129 frames at 720p, resulting in 4x higher energy than Wan 2.1 14B (13B vs. 14B).

We used default runtime parameters (denoising steps, resolution, frames) for all models.
Many of these parameters are **controllable by users**, enabling navigation of the time-energy-quality tradeoff space.
We explored this in a previous benchmark release.

!!! Takeaway
    Diffusion model energy depends on more than model size: number of denoising steps, output resolution, and frame count matter as much or more.
    Video generation can consume one to two orders of magnitude more energy than image generation.


## Deeper Dive into Energy

In this section, we measure and observe how different factors affect energy consumption.

### Batch Size

<figure markdown>
  ![Batch size effect](assets/ml-energy-leaderboard-v3.0/section2-1-batch-size-light.svg#only-light)
  ![Batch size effect](assets/ml-energy-leaderboard-v3.0/section2-1-batch-size-dark.svg#only-dark)
  <figcaption>Energy per token, throughput, median ITL, and power trends against batch size for (a) DeepSeek R1 (Problem Solving) on 8x B200 and (b) Qwen 3 Coder 30B A3B (Code Completion) on 1x B200. Metrics normalized to % of maximum, except power which is normalized to % of GPU TDP.</figcaption>
</figure>

The figure above shows the impact of batch size on energy per token, token generation throughput, median Inter-Token Latency (ITL), and GPU power draw.
Computing hardware typically achieves peak energy efficiency when fully utilized (the [Static Power Wastage](#static-power-wastage) section will go deeper into this).
Therefore, as batch size increases, energy per token drops at first, then plateaus as GPU utilization approaches saturation.

However, the energy efficiency gains of increasing batch size are not without tradeoffs.
Latency (median ITL in this analysis) increases with batch size, as there is strictly more work to do for each batch.
Throughput also increases with batch size, but with diminishing returns as GPU utilization reaches saturation.
Finally, power draw increases with batch size, as a larger portion of the GPU's compute and memory circuitry is actively utilized and drawing power.

From energy per token trends, we can see that DeepSeek R1 has not saturated GPU utilization even at the largest batch size that fits in memory, whereas Qwen 3 Coder approaches saturation around batch size 512.
This explains the two models' throughput trends as well: DeepSeek R1 has a linearly increasing token throughput with batch size as GPU utilization keeps improving, whereas Qwen 3 Coder sees diminishing returns as it approaches saturation.
We can see that these metrics move in tandem rather than in isolation, because they are all heavily coupled with latent factors like GPU utilization.

!!! Takeaway
    Increasing batch size increases latency, power, and throughput, but can unlock 3-5x energy per token reduction.


### Model Size and Architecture

With the Mixture-of-Experts (MoE) architecture, the number of active parameters is as important as the total number of parameters in energy consumption.

<figure markdown>
  ![MoE energy efficiency](assets/ml-energy-leaderboard-v3.0/section2-2-model-size-light.svg#only-light)
  ![MoE energy efficiency](assets/ml-energy-leaderboard-v3.0/section2-2-model-size-dark.svg#only-dark)
  <figcaption>Energy/token by active parameters of Problem Solving models with the minimum-energy configuration on B200 GPUs.</figcaption>
</figure>

The figure above compares models from the Qwen 3 family on the Problem Solving task using B200 GPUs: two MoE variants (30B A3B and 235B A22B) and three dense variants (8B, 14B, and 32B).
For dense models, energy per token increases with the total number of parameters.
However, when we include MoE models, we see that their energy per token is much lower than what a dense model of similar total number of parameters would consume.
For instance, the energy per token of 30B A3B is 3.56x lower than that of 32B, despite having a similar total number of parameters.
However, this is not to say that active parameters are now the only factor.
235B A22B consumes more energy than 32B as it needs to use more GPUs to fit all parameters in GPU memory, though it is still far less than what a dense 235B model would consume.

!!! Takeaway
    MoE models consume less energy compared to dense models of similar total number of parameters, making active parameters an important property for energy consumption.
    However, total parameters, which affect memory requirements, still play a role.


### GPU Generation

One way to compare GPU models (B200 vs. H100) is to pick the minimum-energy configuration on each GPU at the same latency constraint.

<figure markdown>
  ![B200 vs H100](assets/ml-energy-leaderboard-v3.0/section2-3-b200-vs-h100-light.svg#only-light)
  ![B200 vs H100](assets/ml-energy-leaderboard-v3.0/section2-3-b200-vs-h100-dark.svg#only-dark)
  <figcaption>B200 vs H100 energy comparison at latency constraints of 100 ms median ITL for LLMs and 30 s generation latency for Text to Image. Percentage of B200 energy reduction is annotated.</figcaption>
</figure>

Energy reduction can vary significantly by model and task.
Sometimes it is significantly better (e.g., 82% energy per token reduction for Qwen 3 235B A22B Thinking on Problem Solving), other times marginal or even worse, as we will see below.

To get a better overall picture, we compare the two GPU models with three different latency constraints: 50/100/250 ms median ITL for LLMs, 10/30/60 s generation latency for Text to Image, 100/500/1000 s for Text to Video.

**LLM.**
Across all three median ITL constraints, B200 wins 88% (63/72) of comparisons with a median 35% energy reduction (ranging from 53% more to 82% less).
A few notable exceptions happen at tight latency constraints.
B200's large VRAM allows fitting large models on fewer GPUs, avoiding inter-GPU communication overhead.
However, at tight latency constraints, using more H100 GPUs with a higher degree of parallelism can be more energy efficient.
For example, at the 50 ms constraint, Qwen 3 30B A3B Thinking uses 53% less energy on 2x H100 (batch size 128) than on 1x B200 (batch size 64).
Similarly, Qwen 3 235B A22B Instruct FP8 uses 33% less energy on 8x H100 (batch size 192) than on 2x B200 (batch size 64).
At relaxed constraints (> 50 ms), B200 wins as communication overhead is smaller and higher batch sizes become feasible.
We will look deeper into multi-GPU scaling in the [Multi-GPU Scaling](#multi-gpu-scaling) section.

**Diffusion.**
For Text to Image, across all three latency constraints, B200 wins 86% (18/21) of comparisons with a median 15% energy reduction (ranging from 4% more to 23% less).
Text to Video is also similar, with B200 winning 79% (11/14) of comparisons with a median 4% energy reduction (ranging from 6% more to 8% less).
Cases where H100 wins (e.g., Stable Diffusion 3.5 Medium) are generally when the model is small enough to comfortably fit in one H100 GPU, meaning that it will underutilize a B200.

We performed matched latency constraint comparisons, but we note that B200 would be capable of delivering lower latency than H100 when energy is not a concern due to its higher compute and memory throughput.

!!! Takeaway
    B200 achieves lower energy than H100 in 79-88% of comparisons at matched latency constraints.
    For tight LLM latency constraints, H100 can sometimes consume less energy by using more GPUs with higher parallelism to reduce latency.
    For Diffusion, B200 generally wins, unless the model is small.


### Precision

FP8 quantization reduces model memory footprint and allows inference to leverage FP8 Tensor Cores with higher compute throughput.
However, it also adds overhead from extra operations like input/activation quantization, dequantization, and scaling.
We observe this tradeoff playing out differently at different batch sizes.

<figure markdown>
  ![Precision comparison](assets/ml-energy-leaderboard-v3.0/section2-4-precision-light.svg#only-light)
  ![Precision comparison](assets/ml-energy-leaderboard-v3.0/section2-4-precision-dark.svg#only-dark)
  <figcaption>Qwen 3 235B A22B (Text Conversation) on 8x H100. FP8 loses at batch size 8-16, then wins at batch sizes from 32. The dashed vertical lines mark the crossover point.</figcaption>
</figure>

**FP8 wins at larger batch sizes.**
The figure above shows the energy per token and median ITL of Qwen 3 235B A22B (Text Conversation) on 8x H100 in both BF16 and FP8 across batch sizes.
At smaller batch sizes, FP8 loses on both energy and latency due to (1) the overhead of extra operations (especially those that have not been fused into matrix multiplication), and (2) underutilization of the GPU, which prevents FP8 from leveraging its compute throughput advantage.
If we compare FP8 and BF16 for all other models and tasks, we see a similar trend:

<center><b>Energy</b></center>

| Batch size | FP8 wins | Range | Median |
|:-----------|:---------|:------|:-------|
| 8-16 | 0/7 | +13 to +56% | +30% |
| 17-64 | 6/13 | -12 to +32% | +1% |
| 65-256 | 11/12 | -29 to 0% | -11% |

<center><b>Latency</b></center>

| Batch size | FP8 wins | Range | Median |
|:-----------|:---------|:------|:-------|
| 8-16 | 1/7 | -5 to +26% | +7% |
| 17-64 | 11/13 | -23 to +12% | -12% |
| 65-256 | 11/12 | -18 to +3% | -11% |

At batch size 8-16, FP8 has higher energy (up to 56% more) and higher latency (up to 26% slower).
As we grow batch size, we see FP8 starting to win on latency earlier, and then on energy as well.
This is, at least in part, because GPUs are capable of delivering more theoretical FP8 compute throughput than BF16.
Thus, at the same batch size, FP8 underutilizes the GPU more, leading to higher energy consumption until batch size is large enough to saturate the GPU.

**Qwen 3 Coder 480B A35B.**
This model is an exception; due to a limitation in vLLM at the time of benchmarking, the FP8 model had to run attention with data parallelism, while BF16 could run attention with tensor parallelism.[^vllm-qwen-3-coder-fp8-dp]
This made FP8 consistently consume more time and energy across all batch sizes.
Attention data parallelism incurs load imbalance between GPUs that are assigned very different sequence lengths (e.g., some running long prefills whereas others run decode).
Since the straggler GPU bottlenecks the entire batch, this can lead to significant latency overhead.
Furthermore, the non-straggler GPUs do nothing and waste static power (see [Static Power Wastage](#static-power-wastage)) waiting for the straggler, leading to even higher energy consumption as well.

[^vllm-qwen-3-coder-fp8-dp]: Standard parallelization methods used for LLMs: MoE models use expert parallelism with attention tensor parallelism; dense models use Tensor Parallelism for both MLP and attention. For Qwen 3 Coder 480B A35B FP8, see [vLLM Recipes](https://github.com/vllm-project/recipes/blob/a86549479f2c38ac20b96483e7aacd128e3a40b2/Qwen/Qwen3-Coder-480B-A35B.md#fp8-models); last accessed 2024-12-26.

!!! Takeaway
    At smaller batch sizes (8-16), FP8 can consume more time and/or energy than BF16.
    FP8 gains start to appear at larger batch sizes.


### Multi-GPU Scaling

We can execute the same model on different numbers of GPUs, which affects both latency and energy consumption.

<figure markdown>
  ![Multi-GPU scaling](assets/ml-energy-leaderboard-v3.0/section2-5-multi-gpu-light.svg#only-light)
  ![Multi-GPU scaling](assets/ml-energy-leaderboard-v3.0/section2-5-multi-gpu-dark.svg#only-dark)
  <figcaption>Time-energy tradeoffs of GPT-OSS 120B (Problem Solving). In both cases, scaling from 1 GPU to 2 GPUs at fixed batch size trades energy for time. In (b), 1 GPU is limited to batch size 64, while 2 GPUs unlock batch size 2,048 with less energy.</figcaption>
</figure>

The figure above shows GPT OSS 120B on B200 and H100 with 1 and 2 GPUs.
The plots are time-energy tradeoff curves, which are useful in comparing different configurations:

- The right-end of each curve represents the minimum-energy configuration for that GPU model and count.
- A vertical line at one's target latency finds minimum-energy configurations that meet the latency constraint.
- Jumping between curves following points with the same batch size shows the effect of GPU model and count.

**At the same batch size, more GPUs trade energy for latency.**
In general, increasing parallelism with more GPUs reduces latency but also increases energy at the same batch size because (1) latency does not decrease linearly due to communication overhead, and (2) less compute per GPU can lead to lower GPU utilization.
Across B200 configurations, adding GPUs at the same batch size *always* increases energy per token and reduces latency in 81% of cases.
Similarly, across H100 configurations, energy increases in 93% of the cases and latency *always* decreases.

**Memory capacity-bound cases unlock energy savings with more GPUs.**
On top of the above, in cases where adding more GPUs *enables* larger batch sizes due to increased aggregate memory capacity, we can see energy reductions.
For GPT OSS 120B on 1x B200 with a 180 GB VRAM, the model already fits at high batch sizes on 1 GPU (batch size 3,072), so 2 GPUs only add overhead without enabling lower energy.
On 1x H100 with an 80 GB VRAM, however, the server is limited to batch size 64, while 2 GPUs unlock batch size 2,048 and achieve 68% lower minimum energy.
Thus, the model's total parameter memory footprint relative to the GPU's memory capacity is an important factor for whether multi-GPU scaling can reduce energy.

**Case study: Qwen 3 235B A22B Thinking FP8 on Problem Solving.**
As an extra case study, it is interesting to examine Qwen 3 235B A22B Thinking FP8 on Problem Solving with time-energy tradeoff frontiers for four sets of configurations (2x and 4x B200, 2x and 8x H100).

<figure markdown>
  ![Time-energy tradeoff](assets/ml-energy-leaderboard-v3.0/section2-5-235b-tradeoff-light.svg#only-light)
  ![Time-energy tradeoff](assets/ml-energy-leaderboard-v3.0/section2-5-235b-tradeoff-dark.svg#only-dark)
  <figcaption>Time-energy tradeoff for Qwen 3 235B A22B Thinking FP8 on Problem Solving across B200 and H100 with different GPU counts. Each point is annotated with its batch size.</figcaption>
</figure>

The 4x B200 curve (blue) Pareto-dominates, and also achieves the lowest possible energy (~0.4 J/token) by unlocking large batch sizes.
2x B200 (red) consumes less energy per token compared to 4x B200 (blue) at the same batch size at the cost of higher latency (as expected), and fails to scale to large batch sizes due to limited memory capacity.
The two H100 configurations (purple and green) are right in the middle of the B200 curves; despite being a whole generation older, H100 is still competitive!

!!! Takeaway
    At the same batch size, more GPUs typically reduce latency but increase energy.
    When adding GPUs enables larger batch sizes, energy can be reduced, but only if serving was previously limited by memory capacity.
    H100 can still be competitive with B200 in terms of energy, especially when latency constraints are tight.


## Reasoning about Energy Consumption

In the previous sections, we presented empirical observations on energy consumption, but how can we act on them?
In this section, we outline core mechanisms that govern energy consumption, with the goal of providing tools to *explain and reason about* energy consumption.

### Model, Runtime, and Hardware Factors

Many factors across the whole system (hardware, software, and algorithm) affect energy consumption.
Some of the key mechanisms are powerful but still straightforward.
For instance, more computation generally means more energy consumption.
As we have seen, diffusion models' energy increases with more denoising steps and higher output resolution ([Diffusion Models](#diffusion-models)), MoE models activate fewer parameters per token than dense models ([Model Size and Architecture](#model-size-and-architecture)), and FP8 reduces circuit activity via lower-precision arithmetic ([Precision](#precision)).
These are examples of choices at the runtime- and model-level directly affecting the amount of computation, and thus energy consumption.

Another instance is hardware efficiency improvements over generations.
Newer architectures typically deliver more operations per joule via various microarchitectural improvements and technology node shrinks.
We have indeed seen that B200 generally consumes less energy than H100 ([GPU Generation](#gpu-generation)).

### Static Power Wastage

The power consumption of computing hardware, including GPUs, has two components: *static power* (consumed regardless of activity at all times) and *dynamic power* (reflects compute and memory activity).
Let us consider a case where we executed some computation on a GPU, and only 60% of the GPU's compute units were utilized over the entire execution time.
Here, the GPU will consume static power for the entire execution time, regardless of how well the GPU is utilized.
Thus, 40% of the time the GPU is consuming static power while making little progress, effectively wasting energy.
This is how low utilization increases static power wastage and thus energy consumption for the same amount of work.

However, one of the most critical factors in GPU utilization is, in fact, not the GPU, but the rest of the system.
That is, we want the GPU to be the sole bottleneck, not other system components.
When CPU processing, network communication, disk I/O, or other parts of the system block GPU progress, the GPU does not have enough work to saturate itself or is even idle, wasting static power.
Multimodal LLMs ([Multimodal LLM](#multimodal-llm)) were a prime example: CPU-side vision preprocessing became a bottleneck that limited batch size, leaving the GPU underutilized despite having capacity for more concurrent requests.
The result was higher energy per token---not because of the GPU, but because of the surrounding system.

Another important factor is arithmetic intensity, i.e., the ratio of compute operations to the amount of memory movement.
When arithmetic intensity is low, the GPU may be waiting on memory fetches more often than performing computations, leading to lower GPU utilization and higher static power wastage.
We observed this for precision ([Precision](#precision)), where FP8 computations require extra operations that are not as arithmetically intensive as matrix multiplications.
Thus, on smaller batch sizes, both FP8 extra operations and the smaller matrix multiplications had lower arithmetic intensity, leading to lower GPU utilization and offsetting savings from lower-precision arithmetic.

This has interactions with earlier factors as well.
For instance, upgrading to a newer hardware generation expecting better energy efficiency may not yield the expected benefits, or even worsen, if bottlenecks in the rest of the system were preventing the GPU from being fully utilized.


### Time-Energy Tradeoff Frontier

There are many cases where there is a time-energy tradeoff frontier for the same amount of work.
When the GPU ideally *is* the bottleneck, largely ruling out static power wastage ([Static Power Wastage](#static-power-wastage)), we can navigate the time-energy tradeoff frontier through configuration choices.[^tradeoff-note]
In our analysis, the factors that govern this frontier are:

- **Batch size:** This is the primary knob that *shapes* and *navigates* the time-energy frontier.
- **Memory capacity:** Larger batches consume more memory. When GPU memory is saturated, we hit a ceiling, like we have seen for reasoning models in the [LLM](#llm) section. In other words, memory capacity *bookends* the frontier.
- **Application constraints:** Applications may come with latency deadlines or energy budgets. Larger batches increase per-request latency and reduce energy per work. Application-level latency and/or energy budgets allow us to *select* a point on the frontier.

Batch size does not have to be the only knob that shapes the time-energy tradeoff frontier.
For instance, the number of GPUs ([Multi-GPU Scaling](#multi-gpu-scaling)) can be effective, where adding GPUs increases aggregate memory capacity and also enables larger batch sizes that were not previously possible.
While not explored in this post, GPU power limit and core frequency are also knobs that shape the frontier.

[^tradeoff-note]: When the GPU is being underutilized, a proper time-energy *tradeoff* frontier may not exist, as both time and energy can be reduced by improving GPU utilization.


### Extending to AI Datacenters

Our analysis so far has focused on energy consumption, but power is also an important metric to consider.
Indeed, many AI datacenters today are **power-constrained**.
Power availability caps the datacenter's power budget---either from the electricity grid (where drawing too much may not be approved or may cause reliability issues) or from on-site generation like natural gas and batteries (which take *years* to build).

With power becoming the bottleneck resource, **throughput per watt** (e.g., tokens per second per watt, images per second per watt) is a critical metric for AI datacenter operators.
For instance, tokens per second per watt can tell the operator how many average ChatGPT users the datacenter can serve within its power budget.

$$
\text{Throughput per Watt} = \frac{\text{Throughput}}{\text{Power}} = \frac{\text{Work} / \text{Time}}{\text{Energy} / \text{Time}} = \frac{\text{Work}}{\text{Energy}}
$$

Throughput per watt is essentially the inverse of energy consumption per fixed work (e.g., energy per token, energy per image).
Thus, optimizing energy consumption for the given work improves throughput per watt, closing the reasoning loop.

<figure markdown>
  ![Power and energy](assets/ml-energy-leaderboard-v3.0/section3-power-light.svg#only-light)
  ![Power and energy](assets/ml-energy-leaderboard-v3.0/section3-power-dark.svg#only-dark)
  <figcaption>Energy and throughput/watt for four models on B200 with varying batch size. Note the log scale Y axis in (a).</figcaption>
</figure>

As batch size increases, energy per token decreases and throughput per watt increases, both eventually plateauing as the GPU reaches saturation.


### Putting Everything Together

<figure markdown>
  ![Reasoning framework](assets/ml-energy-leaderboard-v3.0/section3-reasoning-framework-light.svg#only-light)
  ![Reasoning framework](assets/ml-energy-leaderboard-v3.0/section3-reasoning-framework-dark.svg#only-dark)
  <figcaption>A framework for reasoning about inference energy consumption in our analysis. Gray boxes are properties and knobs, blue boxes are latent factors, and orange boxes are the end metrics we observe from measurements and would like to understand and explain.</figcaption>
</figure>

The figure above summarizes the structure of reasoning we have developed.
Gray boxes are *properties* and *low-level knobs* of the algorithm, software, and hardware.
Blue boxes represent *latent* variables that mediate between configurations and outcomes.
Orange boxes show the end metrics we measure from benchmarks and ultimately want to understand.

Causal structures like this show how different factors interact and propagate to affect the end metric and provide a framework for explaining empirical observations.
When we observe unexpected energy behavior, we can trace through these factors to identify the root cause---whether it is memory constraints limiting batch size, CPU bottlenecks causing GPU underutilization, or compute volume increasing due to model choices.
This also enables reasoning about optimization opportunities and hypothesizing about how they will affect energy consumption.

!!! Takeaway
    Energy consumption is governed by computation amount, hardware efficiency, GPU utilization, and the time-energy tradeoff frontier.
    System design choices---eliminating non-GPU bottlenecks and navigating the frontier via batch size and memory capacity---are key levers for optimization.
    Throughput per watt is the inverse of energy per output; optimizing one optimizes the other.


## Summing Up

Each section already has key takeaways boxed.

Things are more dynamic, complex, and nuanced than ever before in ML energy consumption. Without even thinking about energy, *time* itself is already so; throwing in energy clearly doesn't help!

Therefore, **we must measure**. If you take away only one thing from this post, let it be that. Back-of-the-envelope estimates and rules of thumb only get you so far.

And, it is very much possible to measure! We hope [The ML.ENERGY Benchmark](https://github.com/ml-energy/benchmark) is a useful tool for you to measure, understand, and ultimately optimize energy consumption in your models, tasks, and systems. And we hope [The ML.ENERGY Leaderboard](https://ml.energy/leaderboard) provides useful reference points for numerous downstream use cases: model selection, system design, deployment planning, policymaking, and more.

Finally, we welcome your feedback, questions, and suggestions. We're always looking forward to hearing from the community and collaborating. Find us at [The ML.ENERGY Initiative](https://ml.energy) homepage.


*[batch size]: The number of requests running concurrently. For LLM and MLLMs with variable number of requests running over time, batch size means the maximum number of requests (`max_num_seqs`) configured for the server. For diffusion models, batch size is exactly the number of items (e.g., images, videos) being generated together.
*[batch sizes]: The number of requests running concurrently. For LLM and MLLMs with variable number of requests running over time, batch size means the maximum number of requests (`max_num_seqs`) configured for the server. For diffusion models, batch size is exactly the number of items (e.g., images, videos) being generated together.
*[minimum-energy configuration]: For a model, task, GPU model, and number of GPUs, selecting the batch size that achieves the lowest energy (e.g., energy per token, energy per response, depending on the context) among all tested batch sizes. This allows us to compare model and task differences without being confounded by hardware utilization differences.
*[minimum-energy configurations]: For a model, task, GPU model, and number of GPUs, selecting the batch size that achieves the lowest energy (e.g., energy per token, energy per response, depending on the context) among all tested batch sizes. This allows us to compare model and task differences without being confounded by hardware utilization differences.
