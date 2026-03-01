---
date: 2026-02-26
authors:
  - ruofanw
  - jaywonchung
categories:
  - energy
  - measurement
links:
  - Kareus paper: https://arxiv.org/abs/2601.17654
  - ZeusMonitor: https://ml.energy/zeus/reference/monitor/#zeus.monitor.ZeusMonitor
---

# Thermally Stable Profiling for Accurate GPU Energy Measurement

Profiling is only as useful as it is accurate.
For example, when an optimizer evaluates candidate configurations, biased or noisy measurements can steer it toward the wrong answer.
In this post, we describe a GPU energy profiling methodology adopted in [Kareus](https://arxiv.org/abs/2601.17654), and show experimentally why it matters and what we found to be the right settings.

<!-- more -->

## The Problem

We use [Zeus](https://ml.energy/zeus) (which internally uses [NVML](https://docs.nvidia.com/deploy/nvml-api/)) to measure the time and energy of GPU workloads.
At first glance this seems straightforward: start a measurement window, run the workload, stop the window, read off the energy.
But two factors conspire to make this unreliable.

**NVML's sampling granularity.**
NVML updates its energy counters at a coarse granularity — approximately every 100 ms on NVIDIA GPUs.[^1]
When dealing with short workloads, such as a handful of kernels that finishes in milliseconds, the only practical solution is to execute the workload repeatedly within a measurement window and report the average energy per run.
Even then, short measurement windows — such as one second — remain vulnerable to "sampling noise." Because there are so few updates in the time span, those discrete jumps can heavily distort the final energy reading.

[^1]: The 100 ms counter update interval is observable in the [NVIDIA open GPU kernel module source](https://github.com/NVIDIA/open-gpu-kernel-modules).

**Temperature-dependent power draw.**
GPU power consumption is temperature-dependent.
When you run a series of profiling trials back-to-back, each trial heats the GPU up.
The next trial then starts from an elevated temperature and draws more power, producing a systematically higher energy reading.
If you're comparing two configurations and one happens to run after a hotter predecessor, the comparison is unfair.

We address both with what we call *thermally stable profiling*: a measurement window long enough for stable readings, and a cooldown period between trials to reset the GPU's temperature.

## Design Choices

**Measurement window.**
We execute each workload repeatedly over a 5-second window rather than timing a single run.
A warm-up pass first determines execution time; from that, we compute the number of iterations needed to fill 5 seconds.

**Thermal cooldown.**
We insert a 5-second pause between consecutive profiling trials.
In our environment, this reliably brings the GPU below 32°C before the next trial begins.
Note that the right duration depends on the server's cooling capability.

## Experimental Analysis

To quantify how much these choices matter, we ran a controlled experiment using the Attention layer of Llama 3.2 3B on 8 NVIDIA A100 GPUs.
For each configuration, we repeated the profiling trial 10 times and report the distribution of measured energy consumption.

<figure markdown>
<div style="display: flex; gap: 2%;" markdown>
<div style="flex: 1; min-width: 0;" markdown>

![Measurement duration](assets/thermally-stable-profiling/duration_energy_plot-light.svg#only-light)
![Measurement duration](assets/thermally-stable-profiling/duration_energy_plot-dark.svg#only-dark)

</div>
<div style="flex: 1; min-width: 0;" markdown>

![Cooldown duration](assets/thermally-stable-profiling/cooldown_energy_plot-light.svg#only-light)
![Cooldown duration](assets/thermally-stable-profiling/cooldown_energy_plot-dark.svg#only-dark)

</div>
</div>
<figcaption>Impact of changing (a) measurement duration and (b) cooldown duration. We report the distribution of energy across 10 repeated trials, along with the average GPU temperature before and after measurement.</figcaption>
</figure>

### Measurement Duration

We fix the cooldown to 5 seconds and sweep the measurement window.

When the measurement window is short (e.g., less than 2 seconds), the reported energy consumption exhibits large variability. 
This is attributed to the 100 ms NVML counter update interval, which introduces discretization noise in short-duration measurements. 
In addition, shorter windows yield lower average energy readings because the GPU has not fully warmed up.
Energy measurement stabilizes from 5 seconds onward, which is why we chose 5 seconds as our measurement duration.

!!! Takeaway
    Measurement windows shorter than ~5 seconds exhibit high variability due to NVML's 100 ms sampling granularity. Energy measurement stabilize from 5 seconds onward.

### Cooldown Duration

We fix the measurement window to 5 seconds and sweep the cooldown duration between consecutive trials.

We observe that average energy consumption strongly correlates with the GPU temperature at the start of measurement.
Without a cooldown period, each trial inherits a hotter GPU from the previous run, resulting in biased and high-variance energy measurements.
With 5 seconds of cooldown, the GPU returns to a stable thermal baseline before each trial.
Beyond this point, the measured energy consumption stabilizes, so we adopt a 5-second cooldown duration.

!!! Takeaway
    Without thermal cooldown between trials, measured energy correlates strongly with GPU temperature at the start of measurement. A 5-second cooldown brings the GPU to a consistent baseline and eliminates this bias.

## Summing Up

Accurate GPU energy profiling requires attending to two things: giving the measurement window enough time for NVML counter statistics, and resetting the GPU's thermal state between consecutive trials.
In our setup, 5 seconds for the measurement window and 5 seconds for cooldown are sufficient.
The right numbers will vary by GPU firmware and server cooling capability, but the experimental methodology generalizes to any environment.
