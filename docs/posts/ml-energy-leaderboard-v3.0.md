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
---

# Reading the ML.ENERGY Leaderboard v3.0

With [The ML.ENERGY Benchmark v3.0](https://github.com/ml-energy/benchmark/releases/tag/v3.0) we released in December 2025, we expanded our scope to up-to-date important models, tasks, and GPU hardware.
This included 46 models across 7 tasks, producing 1,858 configurations on NVIDIA H100 and B200 GPUs.[^software-setup]
As always, latest benchmarking results are public and can be browsed on [The ML.ENERGY Leaderboard](https://ml.energy/leaderboard).
This post presents notable results from the v3.0 benchmark run.

<!-- more -->

For more details on our methodology, please refer to [our NeurIPS 25 D&B paper](https://arxiv.org/abs/2505.06371).

[^software-setup]: We used vLLM 0.11.1 for LLM/MLLMs and xDiT 0.4.5 for diffusion models.

## Energy by Architecture

What determines the energy consumption of generating one **response**?
For Large Language Models (LLMs), a response is a complete answer to a prompt with all output tokens included.
For diffusion models, a response is one generated image or video.

### LLM

For LLMs, energy per response is simply energy per token multiplied by the number of output tokens.
The number of output tokens varies significantly by the model and task type.

**Task type heavily influences output length.**
Different tasks naturally produce different output lengths (number of output tokens).
This is particularly pronounced between two LLM tasks in our benchmark: Problem Solving (reasoning on) and Text Conversation (reasoning off).

<figure markdown>
  ![Energy and output length distributions](assets/ml-energy-leaderboard-v3.0/section1-1-llm-light.svg#only-light)
  ![Energy and output length distributions](assets/ml-energy-leaderboard-v3.0/section1-1-llm-dark.svg#only-dark)
  <figcaption>Distribution of output length, energy per token, and energy per response across models. Each data point is the model's minimum-energy configuration on B200 GPUs.</figcaption>
</figure>

Here, we're comparing the minimum-energy configuration of each model on B200 GPUs, which allows us to focus on model and task differences without being confounded by hardware utilization differences.
Problem Solving on average generates 10x more output tokens than Text Conversation (mean 6,988 vs. 717).
On top of this, longer output sequences stress memory capacity and prevent larger batch sizes, which increases energy per token due to lower GPU utilization.
Therefore, Problem Solving responses consume on average 25x more energy than Text Conversation (mean 4,625 J vs. 184 J).

**Case study on Qwen 3 32B on 1x B200.**
Qwen 3 32B supports both reasoning mode and non-reasoning mode, allowing direct comparison of energy consumption for the same model on different tasks.

| Metric | Problem Solving | Text Conversation | Ratio |
|--------|----------------:|------------------:|-------|
| Max batch size | 128 | 512 | 4x lower |
| Average output tokens | 7,035 | 627 | 11x more |
| Energy/token @ batch size 128 | 0.312 J | 0.209 J | 1.5x higher |
| Energy/token @ max batch size | 0.312 J | 0.151 J | 2.1x higher |
| Energy/response | 2,192 J | 95 J | 23x more |

Longer output sequences in Problem Solving increases KV cache size, preventing the server from running larger batch sizes.
Therefore, when we compare energy per token at each task's maximum batch size, Problem Solving is 2.1x higher.
Even at the same batch size 128, longer sequences consumes more energy per token due to higher memory footprint.
Finally, combining longer outputs and higher energy per token results in 23x more energy per response for Problem Solving.

We do want to note that one factor we're not comparing here is the number of input/prompt tokens (mean 634 vs. 224).
While there is a difference, both are not considered very long input sequences, and we do not expect this to have visible effect, particularly when inference iterations are entirely dominated by decode rather than prefill.

!!! Takeaway
    Task type heavily influences energy consumption.
    Notably, Problem Solving uses on average 25x more energy per response than Text Conversation.
    This comes from 10x more output tokens combined with higher energy per token due to memory pressure from long sequences.


### Multimodal LLM

Multimodal LLMs (MLLMs) takes images and/or videos alongside text as input and generates text responses.

<figure markdown>
  ![MLLM energy by modality](assets/ml-energy-leaderboard-v3.0/section1-2-mllm-light.svg#only-light)
  ![MLLM energy by modality](assets/ml-energy-leaderboard-v3.0/section1-2-mllm-dark.svg#only-dark)
  <figcaption>Energy per token by input modality (left), and Qwen 3 (VL) 8B batch size and GPU KV cache utilization (right) on B200. Minimum-energy configurations are used.</figcaption>
</figure>

**Multimodality can increase energy.**
The implications of multimodal inputs are threefold:

1. The model needs to run its vision encoder to convert them into vision tokens, which can sometimes increase input length significantly and increase computation and memory usage.
2. In GPU memory-constrained scenarios, the memory consumption of the vision encoders (weight and activation) can further limit batch size.
3. The input image or video needs to be preprocessed first on the CPU-side (e.g., converting raw image/video into tiles of pixels), which can take non-negligible time and become a bottleneck that limits batch size.

Indeed, when we compare minimum-energy configurations for the same model family fixing the GPU model and number of GPUs, processing images uses 1.1-5.2x the energy per token of text, while video uses 1.3-15.0x.[^mllm-task-output-length]

**Case study on Qwen 3 (VL) 8B on 1x B200.**
We're comparing Qwen 3 8B on Text Conversation with Qwen 3 VL 8B on Image Chat and Video Chat.
For this smaller model, the overheads of vision encoders and CPU-side preprocessing limit batch size significantly and underutilizes the GPU.
Especially, video inputs typically get converted to more vision tokens and are more expensive to preprocess from the CPU-side, which shows from the much smaller batch size and higher energy per token.
The drop in GPU KV cache utilization as vision preprocessing overhead grows larger confirms that GPU memory had capacity to spare longer prompts from vision tokens, but CPU-side vision preprocessing became a severe bottleneck that limited batch size.

All in all, this is a case where GPU energy consumption is not just about the GPU; the entire system and where the bottlenecks are matter a lot.[^gpu-multimodal-processing]
If CPU-side processing speed is similar and just the GPU is upgraded, GPUs will only be more underutilized.

[^mllm-task-output-length]: One caveat of this cross-modality comparison is that, as we have seen in the LLM section, different tasks can have different output lengths that affect energy per token. For the models we compared, Text Conversation, Image Chat, and Video Chat have average output lengths of 808, 944, and 392 tokens, respectively. This isn't as large as the difference between Text Conversation and Problem Solving and shouldn't affect energy per token as much as that case, but Video Chat's shorter output length (which allows requests to finish faster and reduces batch size when the CPU is the bottleneck) may have increased energy per token compared to Image Chat and Text Conversation.

[^gpu-multimodal-processing]: There are some proposals to run vision preprocessing on the GPU itself (e.g., [vLLM #21995](https://github.com/vllm-project/vllm/issues/21995)), which can alleviate CPU-side bottlenecks but instead shift compute more to the GPU, which will likely introduce its own interesting tradeoffs.

!!! Takeaway
    Multimodal inputs cost 1.1-5.2x (image) and 1.3-15.0x (video) the energy per token of text at the same GPU count.
    CPU-side vision preprocessing is a bottleneck that reduces batch size and increases energy per token.


### Diffusion Models

We benchmarked diffusion models that generate images and videos from user text prompts.
Diffusion is where model size is not the best predictor of energy consumption due to multiple *runtime* factors: number of inference (denoising) steps, output resolution, and number of frames (for video).

<figure markdown>
  ![Text-to-image energy](assets/ml-energy-leaderboard-v3.0/section1-3-diffusion-light.svg#only-light)
  ![Text-to-image energy](assets/ml-energy-leaderboard-v3.0/section1-3-diffusion-dark.svg#only-dark)
  <figcaption>Energy per image/video for diffusion models (minimum-energy configuration on B200). SD is short for Stable Diffusion.</figcaption>
</figure>

**Text-to-image varies 20x across models.**
Models and runtime parameters range from 0.6B to 12B parameters and 20-50 denoising steps.
Notably, Hunyuan-DiT 1.2 (1.5B parameters) consumes more energy than SD 3.5 Large (8.1B parameters) despite having fewer parameters, due in large part to running 50 denoising steps versus 28.

**Text-to-video can be very energy intensive.**
Generating a single video consumes 26 kJ to 1.16 MJ; one to two orders of magnitude more energy than images.
Regarding factors, output resolution and frame count are also important in video generation.
CogVideoX 1.5 5B uses more energy than Wan 2.1 14B despite being a smaller model, largely because it generates at higher resolution (768x1360 vs 480x832).
HunyuanVideo stands out at an extreme 1.16 MJ because it generates 129 frames at a resolution of 720p, which is significantly more total pixels than Wan 2.1 14B (which generates 81 frames at 480p), resulting in 4x higher energy despite similar model sizes (13B vs 14B).

For all models, we used their default runtime parameters.
It's worth noting that many of these runtime parameters are **controllable by users** (e.g., number of denoising steps, output resolution), and allows users to navigate the time-energy-quality tradeoff space.

!!! Takeaway
    Diffusion model energy depends on more than model size: number of denoising steps, output resolution, and frame count matter as much or more.
    Video generation can consume one to two orders of magnitude more energy than image generation.


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
    Increasing batch size increases latency, power, and throughput, and can unlock 3-5x energy per token reduction.


### Model Size

With the Mixture-of-Experts (MoE) architecture, the total number of parameters is less of a determinant of energy consumption; the number of *active* parameters is important.

<figure markdown>
  ![MoE energy efficiency](assets/ml-energy-leaderboard-v3.0/section2-2-model-size-light.svg#only-light)
  ![MoE energy efficiency](assets/ml-energy-leaderboard-v3.0/section2-2-model-size-dark.svg#only-dark)
  <figcaption>Energy per token by active parameters for Qwen 3 model variants on B200 (Problem Solving task, minimum-energy configuration).</figcaption>
</figure>

Within the Qwen 3 model family, we compare two MoE variants, 30B A3B (30B total, 3B active) and 235B A22B (235B total, 22B active), and three dense variants, 8B, 14B, and 32B.
For dense models, energy per token increases with model size (total number of parameters).
However, when we throw in MoE models, we see that their energy per token is much lower than what a dense model of similar total size would consume.
For instance, the energy per token of Qwen 3 30B A3B is 3.56x lower than that of Qwen 3 32B, despite having a similar total number of parameters.
Qwen 3 235B A22B consumes more energy than 32B as it needs to use more GPUs to fit all parameters in GPU memory, but still far less than what a dense 235B model would.

!!! Takeaway
    MoE models consume much less energy compared to dense models of similar total number of parameters.
    The number of active parameters is an important factor, though other factors like memory pressure also play a role.


### B200 versus H100

How much is B200 better than H100 in terms of energy?
For each model, let's compare the minimum energy configuration on B200 against H100 that meets latency constraints.

<figure markdown>
  ![B200 vs H100](assets/ml-energy-leaderboard-v3.0/section2-3-b200-vs-h100-light.svg#only-light)
  ![B200 vs H100](assets/ml-energy-leaderboard-v3.0/section2-3-b200-vs-h100-dark.svg#only-dark)
  <figcaption>B200 vs H100 energy comparison at matched latency constraints. Percentage of B200 energy reduction over H100 is annotated on each bar.</figcaption>
</figure>

As shown in the figure, energy reduction can vary a lot by model and task.
Sometimes it's significantly better (e.g., 82% energy per token reduction for Qwen 3 235B A22B Thinking on Problem Solving), other times it could be marginal, or, as we will see below, sometimes worse.

To get a better overall picture, we also compared the two GPU models for all models with three different latency constraints (50/100/250 ms median ITL for LLMs, 10/30/60 s generation latency for text-to-image, 100/500/1000 s for text-to-video).

**LLM.** Across all three median ITL constraints, B200 wins 88% (63/72) of comparisons with a median 35% energy reduction (-53% to +82%).
A few notable exceptions happen at tight latency constraints.
Namely, while B200's large VRAM allows fitting large models on fewer GPUs without inter-GPU communication overhead, at tight latency constraints, using more H100 GPUs with higher degree of parallelism can be more energy efficient.
For example, at the 50 ms constraint: Qwen 3 30B A3B Thinking uses 53% less energy on H100 (2 GPUs, batch size 128) compared to B200 (1 GPU, batch size 64), and Qwen 3 235B A22B Instruct FP8 uses 33% less energy on H100 (8 GPUs, batch size 192) compared to B200 (2 GPUs, batch size 64).
At relaxed constraints (> 50 ms), B200 wins as communication overhead is smaller and higher batch sizes become feasible.
We will look deeper into multi-GPU scaling in a later section.

**Diffusion (Text to Image).** Across all three latency constraints, B200 wins 86% (18/21) of comparisons with a median 15% energy reduction (-4% to +23%).
Stable Diffusion 3.5 Medium is the outlier, but still, H100 wins only by a small margin (by 4%).

**Diffusion (Text to Video).** Across all three latency constraints, B200 wins 79% (11/14) of comparisons with a median 4% energy reduction (-6% to +8%).
In general, we can say that B200 and H100 consume similar energy for video generation.
B200 still provides lower latency, which matters for user experience.

!!! Takeaway
    B200 achieves lower energy in 79-88% of comparisons at matched latency constraints.
    Particularly for LLMs, B200 provides a median 35% energy reduction, but with tight latency constraints, using more H100 GPUs with higher parallelism can consume less energy.


### Precision

FP8 quantization reduces model memory footprint and increases compute throughput with FP8 Tensor Cores.
However, it also adds overhead from the extra operations like dequantization/rescaling.
We see that this tradeoff plays out differently at different batch sizes.

<figure markdown>
  ![Precision comparison](assets/ml-energy-leaderboard-v3.0/section2-4-precision-light.svg#only-light)
  ![Precision comparison](assets/ml-energy-leaderboard-v3.0/section2-4-precision-dark.svg#only-dark)
  <figcaption>FP8 loses at batch size 8-16, then wins at batch sizes from 32. The dashed vertical lines mark the crossover point.</figcaption>
</figure>

**FP8 wins at large batch sizes.**
When we compare models with official weights in both BF16 and FP8 on the same GPU model, number of GPUs, and batch size (excluding the Qwen 3 Coder 480B A35B case discussed below), we see the following:

- **Batch size 8-16:** FP8 wins 0/7 on energy (median +30%, range +13% to +56%), 1/7 on latency (median +7%, range -5% to +26%)
- **Batch size 17-64:** FP8 wins 6/13 on energy (median +1%, range -12% to +32%), 11/13 on latency (median -12%, range -23% to +12%)
- **Batch size 65-256:** FP8 wins 11/12 on energy (median -11%, range -29% to +0%), 11/12 on latency (median -11%, range -18% to +3%)

At batch size 8-16, FP8 has higher energy (up to 56% more) and higher latency (up to 26% slower).
The dequantization and rescaling overhead does not get amortized well at small batch sizes, leading to higher time and energy.
As we grow the batch size, FP8's energy advantage comes more gradually at relatively larger batch sizes than latency.
This is, at least in part, because GPUs are capable of delivering more FP8 compute throughput than BF16 and at the same batch size, FP8 underutilizes the GPU more, leading to higher energy consumption until batch size is large enough to saturate the GPU.

**Qwen 3 Coder 480B A35B.**
In our benchmark, we select the parallelization method for an LLM simply by architecture: MoE models use Expert Parallelism with attention Tensor Parallelism, while other models use Tensor Parallelism for both MLP and attention.
This makes FP8 vs BF16 comparisons for the same model straightforward since they use the same parallelization.
However, Qwen 3 Coder 480B A35B is an exception; due to a limitation in vLLM at the time of benchmarking, the FP8 model had to run attention with data parallelism, while BF16 could run attention with tensor parallelism.[^vllm-qwen-3-coder-fp8-dp]
This made FP8 consistently consume more time and energy than BF16 across all batch sizes.
Attention data parallelism, especially *without* Prefill-Decode disaggregation, incurs load imbalance between GPUs that are assigned very different sequence lengths (e.g., some running long prefills whereas others run decode).
Since the straggler GPU bottlenecks the entire batch, this can lead to significant latency overhead.
Furthermore, the non-straggler GPUs do nothing and waste static power waiting for the straggler, leading to even higher energy consumption as well.

[^vllm-qwen-3-coder-fp8-dp]: [vLLM Recipes](https://github.com/vllm-project/recipes/blob/a86549479f2c38ac20b96483e7aacd128e3a40b2/Qwen/Qwen3-Coder-480B-A35B.md#fp8-models); last accessed 2024-12-26

!!! Takeaway
    FP8 typically wins at batch sizes north of 32 on both energy and latency.
    At batch size 8-16, FP8's fixed overhead makes it slower and less efficient than BF16.


### Multi-GPU Scaling

We can execute the same model on different numbers of GPUs, which affects both latency and energy consumption.

<figure markdown>
  ![Multi-GPU scaling](assets/ml-energy-leaderboard-v3.0/section2-5-multi-gpu-light.svg#only-light)
  ![Multi-GPU scaling](assets/ml-energy-leaderboard-v3.0/section2-5-multi-gpu-dark.svg#only-dark)
  <figcaption>Time-energy tradeoffs of GPT OSS 120B on B200 (left) and H100 (right). In both cases, scaling from 1 GPU (blue line) to 2 GPUs (red line) at fixed batch size trades energy for time. On H100 particularly, 1 GPU is memory-limited to batch size 64, while 2 GPUs unlock batch size 2048, which achieves much lower energy per token.</figcaption>
</figure>

We have drawn a time-energy tradeoff curve, which is useful in comparing different configurations:

- Draw a vertical line at your target latency to find the energy per token of different configurations that meet the latency target.
- Jump between curves following points with the same batch size to see how GPU type and GPU count affect time and energy.
- Find the right-end of each curve to see the minimum-energy configuration for that GPU type and GPU count.


**At the same batch size, more GPUs trade energy for latency.**
In general, increasing parallelism with more GPUs reduces latency but also increases energy at the same batch size because (1) latency does not decrease linearly due to communication overhead, and (2) less compute per GPU can lead to lower GPU utilization.
On B200 configurations, adding GPUs at the same batch size reduced latency in 81% of cases and *always* increased energy per token.
Similarly, on H100 configurations, energy increases in 93% of the cases and latency *always* decreases.

**Energy savings require models that need the extra memory.**
On top of the above, in cases where adding more GPUs *enables* larger batch sizes due to increased aggregate memory capacity, we can see energy reductions.
For GPT OSS 120B on 1x B200 with a 180 GB VRAM, the model already fits at high batch sizes on 1 GPU (batch size 3072), so 2 GPUs only add overhead without enabling lower energy.
On 1x H100 with an 80 GB VRAM, the server is limited to batch size 64, while 2 GPUs unlock batch size 2048 and achieve 68% lower minimum energy.
So, the model's total parameter memory footprint relative to the GPU's memory capacity is an important factor in determining whether multi-GPU scaling can reduce energy by enabling larger batch sizes.

**Case study: Qwen 3 235B A22B Thinking FP8 on Problem Solving.**
As an extra case study, it is interesting to look at Qwen 3 235B A22B Thinking FP8 on Problem Solving with configurations from 2x to 8x GPUs on both B200 and H100 drawn together on the same time-energy tradeoff plot.

<figure markdown>
  ![Time-energy tradeoff](assets/ml-energy-leaderboard-v3.0/section2-5-235b-tradeoff-light.svg#only-light)
  ![Time-energy tradeoff](assets/ml-energy-leaderboard-v3.0/section2-5-235b-tradeoff-dark.svg#only-dark)
  <figcaption>Time-energy tradeoff for Qwen 3 235B A22B Thinking FP8 on Problem Solving across B200 and H100 with different GPU counts. Each point is annotated with its batch size.</figcaption>
</figure>

4x B200 (blue) Pareto-dominates, achieving the lowest possible energy (~0.4 J/token) and unlocking large batch sizes.
2x B200 (red) consumes less energy per token compared to 4x B200 at the same batch sizes (as expected), but at the cost of higher latency and fails to scale to large batch sizes.
The two H100 configurations (purple and green) are right in the middle of the B200 curves; despite being a whole generation older, H100 is still competitive!

!!! Takeaway
    At the same batch size, more GPUs typically reduce latency but increase energy.
    When adding GPUs enables larger batch sizes, energy can be reduced--but only if serving was previously limited by memory capacity.
    H100 can still be competitive with B200 in terms of energy, especially when latency constraints are tight.


## Reasoning about Energy Consumption

In the previous sections, we presented empirical observations on energy consumption.
Now, what do we make of these results?
In this section, we outline core mechanisms that govern energy consumption, with the goal of providing tools to *reason about* energy consumption.

### The Core Mechanisms

Many factors across the whole system (hardware, software, and algorithm) affect energy consumption.
Some of the key mechanisms trivial.
For instance, more computation generally means more energy consumption.
MoE models activate fewer parameters per token than dense models ([Model Size](#model-size)), FP8 reduces computation via lower-precision arithmetic ([Precision](#precision)), and diffusion models' energy scales with denoising steps and output resolution ([Diffusion Models](#diffusion-models)).
This is why model-level efficiency improvements matter greatly.

Another instance is hardware efficiency improvements over generations.
Newer GPU architectures typically deliver more operations per joule.
We've indeed seen that B200 generally consumes less energy than H100 for the same amount of work ([B200 versus H100](#b200-versus-h100)), though the gap varies by model and configuration.

The aforementioned two mechanisms are largely about model and hardware choices.
However, in order to understand and reason about energy consumption in real world systems, we go deeper into system-level mechanisms.

### Static Power Wastage

The power consumption of the GPU, which is the major energy consumer, has two components: *static power* (consumed regardless of activity) and *dynamic power* (reflects compute and memory activity).
When the GPU is underutilized, static power is consumed during that time period anyway, wasting energy and reducing energy efficiency for the same amount of work.
Therefore, higher GPU utilization leads to lower energy for the same amount of work.

One of the most critical factors in GPU utilization is actually not the GPU, but the rest of the system.
That is, the GPU should be the bottleneck, not other system components.
When CPU preprocessing, network communication, or other overheads limit throughput, the GPU sits idle waiting, wasting static power.

We saw this with multimodal LLMs ([Multimodal LLM](#multimodal-llm)): CPU-side vision preprocessing became a bottleneck that limited batch size, leaving GPU memory underutilized despite having capacity for more concurrent requests.
The result was higher energy per token—not because of the GPU, but because of the surrounding system.

This has practical implications: upgrading to a faster GPU without addressing system bottlenecks may actually *worsen* energy per output, as the more powerful GPU spends even more time waiting and wasting static power.


### Time-Energy Tradeoff Frontier

As we have seen, there are cases where there is a time--energy tradeoff frontier for the same amount of work.
When the GPU *is* the bottleneck—the ideal case, we can navigate the time-energy tradeoff frontier through configuration choices.
Which point on the frontier we can choose depends on application-level requirements; it could be latency deadlines for energy budgets.

For many ML applications, batch size is one of the most important lever.
Especially for LLM decode, larger batches increase arithmetic intensity, improving utilization and reducing energy per token ([Batch Size](#batch-size)).
However, batch size is constrained by two factors:

- **Memory capacity:** Larger batches require more memory for activations and KV cache for LLMs. When GPU memory is exhausted, we hit a ceiling. This is why models with longer outputs achieve lower maximum batch sizes ([LLM](#llm)).
- **Latency requirements:** Larger batches increase per-request latency. Strict latency constraints may force smaller batch sizes than what would minimize energy.

Many configurations alter the time-energy tradeoff frontier itself, and [Multi-GPU Scaling](#multi-gpu-scaling) is a prime example.
Adding GPUs increases aggregate memory capacity, potentially enabling larger batch sizes that weren't previously possible ([Multi-GPU Scaling](#multi-gpu-scaling)).
But this only helps if the single-GPU configuration was memory-limited.
Otherwise, additional GPUs just add communication overhead without enabling better configurations.


### Power and Energy

We've briefly touched on what happens to GPU power draw when we change batch size [earlier](#batch-size); it generally increases with batch size as the GPU is more fully utilized.
Is that all there is for power?

Many AI datacenters today are **power-constrained**; they have more hardware inside than their maximum power budget (this is called *oversubscription*).
Especially for AI datacenters with massive power draw, the datacenter's power budget is constrained heavily by power availability -- either from the electricity grid (drawing too much will likely not be approved, could cause grid reliability issues, or result in price hikes), or from on-site power generation like natural gas and batteries (which take minimum *years* to build).

That's why one of the most important metrics today is **throughput per watt** (e.g., tokens per second per watt, images per second per watt).
It tells you how much work you can get done within your power budget.
If it's tokens per second per watt, it tells you how many tokens you can generate per second in your power-constrained datacenter, which translates to how many ChatGPT users you can serve, for instance.

How does this relate to energy?

$$
\text{Throughput per Watt} = \frac{\text{Throughput}}{\text{Power}} = \frac{\text{Work} / \text{Time}}{\text{Energy} / \text{Time}} = \frac{\text{Work}}{\text{Energy}}
$$

So, throughput per watt is basically the inverse of energy consumption per fixed amount of work (e.g., energy per token, energy per image).
Thus, optimizing energy consumption for the given work improves throughput per watt.

<figure markdown>
  ![Power and energy](assets/ml-energy-leaderboard-v3.0/section3-power-light.svg#only-light)
  ![Power and energy](assets/ml-energy-leaderboard-v3.0/section3-power-dark.svg#only-dark)
  <figcaption>Energy per token (left) and token throughput per watt (right) for four models on B200. Note that the left plot's Y axis is log scale.</figcaption>
</figure>

As batch size increases, energy per token decreases and throughput per watt increases, both eventually plateauing as the GPU reaches saturation.

!!! Takeaway
    Energy consumption is governed by four mechanisms: computation amount, hardware efficiency, GPU utilization, and the time-energy tradeoff frontier.
    System design choices—eliminating non-GPU bottlenecks and navigating the frontier via batch size and memory capacity—are key levers for optimization.
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
