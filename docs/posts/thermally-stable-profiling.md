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
  - zeus.profile: http://ml.energy/zeus/reference/profile/#zeus.profile
  - ZeusMonitor: https://ml.energy/zeus/reference/monitor/energy/#zeus.monitor.energy.ZeusMonitor
---

# Thermally Stable Profiling for Accurate GPU Energy Measurement

Profiling is only as useful as it is accurate.
For example, when an optimizer evaluates candidate configurations, biased or noisy measurements can steer it toward the wrong answer.
In this post, we describe the GPU energy profiling methodology adopted in [Kareus](https://arxiv.org/abs/2601.17654), show experimentally why it matters, and introduce the [`zeus.profile`](http://ml.energy/zeus/reference/profile/#zeus.profile) module that automates the process.

<!-- more -->

## The Problem

We use [Zeus](https://ml.energy/zeus) (which internally uses [NVML](https://docs.nvidia.com/deploy/nvml-api/)) to measure the energy consumption of GPU workloads.
At first glance this seems straightforward: start a measurement window, run the workload, stop the window, read off the energy.
But two factors conspire to make this unreliable.

**NVML's sampling granularity.**
NVML updates its energy counters at a coarse granularity, approximately every 100 ms on NVIDIA GPUs.[^1]
When dealing with short workloads, such as a handful of kernels that finishes in milliseconds, the only practical solution is to execute the workload repeatedly within a measurement window and take the average energy per run.
Even then, short measurement windows, such as one second, remain vulnerable to "sampling noise." Because there are so few updates in the time span, those discrete jumps can heavily distort the final energy reading.

[^1]: The 100 ms minimum counter update interval can be found in the [NVIDIA open GPU kernel module source](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/590.48.01/src/nvidia/interface/nvrm_registry.h#L2982).

**Temperature-dependent power draw.**
Hardware power consumption is temperature-dependent; higher temperature primarily leads to increased leakage current and thus higher power draw.
When you run a series of profiling trials back-to-back, each trial heats the GPU up.
The next trial then starts from an elevated temperature and draws more power, producing a systematically higher energy reading.
If you're comparing two configurations and one happens to run after a hotter predecessor, the comparison is unfair.

We address both with what we call *thermally stable profiling*: a measurement window long enough for stable readings, and a cooldown period between trials to reset the GPU's temperature.

## Experimental Analysis

To quantify how much these choices matter, we ran a controlled experiment using the Attention layer of Llama 3.2 3B on 8 NVIDIA A100 GPUs.
For each configuration, we repeated the profiling trial 10 times and report the distribution of measured energy consumption.

<figure markdown>
<div style="display: flex; flex-wrap: wrap; gap: 2%;" markdown>
<div style="flex: 1; min-width: 300px;" markdown>

![Measurement duration](assets/thermally-stable-profiling/duration_energy_plot-light.svg#only-light)
![Measurement duration](assets/thermally-stable-profiling/duration_energy_plot-dark.svg#only-dark)

</div>
<div style="flex: 1; min-width: 300px;" markdown>

![Cooldown duration](assets/thermally-stable-profiling/cooldown_energy_plot-light.svg#only-light)
![Cooldown duration](assets/thermally-stable-profiling/cooldown_energy_plot-dark.svg#only-dark)

</div>
</div>
<figcaption>Impact of changing measurement duration (left) and cooldown duration (right). We report the distribution of energy across 10 repeated trials, along with the average GPU temperature before and after measurement.</figcaption>
</figure>

### Measurement Duration

Rather than measuring a single run, we execute each workload repeatedly over a measurement window: a warm-up pass determines execution time, and from that we compute how many iterations fill the window.
We fix the between-trial cooldown duration to 5 seconds and sweep the measurement window duration.

When the measurement window is short (e.g., less than 2 seconds), the reported energy consumption exhibits large variability. 
This is attributed to the 100 ms NVML counter update interval, which introduces discretization noise in short-duration measurements. 
In addition, shorter windows yield lower average energy readings because the GPU has not fully warmed up.
Energy measurement stabilizes from 5 seconds onward, so we adopt that as our measurement duration.

### Cooldown Duration

Between consecutive profiling trials, we insert a cooldown pause to let the GPU return to a stable temperature before the next measurement.
We fix the measurement window to 5 seconds and sweep the cooldown duration.

We observe that average energy consumption strongly correlates with the GPU temperature at the start of measurement.
Without a cooldown period, each trial inherits a hotter GPU from the previous run, resulting in biased and high-variance energy measurements.
In our environment, 5 seconds of cooldown is enough for the GPU to return to a stable thermal baseline before each trial.
Beyond this point, the measured energy consumption stabilizes, so we adopt a 5-second cooldown duration.

!!! Takeaway
    In this experiment, we settle on a 5-second measurement window (long enough to average out NVML's 100 ms counter granularity) and a 5-second cooldown between trials (enough to bring the GPU back to a stable thermal baseline). The right values depend on your workload, your GPU, and your cooling environment, so these parameters need to be profiled for each setup.

## Try It Out

The [`zeus.profile`](http://ml.energy/zeus/reference/profile/#zeus.profile) module provides functions that sweep candidate measurement and cooldown durations and identify configurations that yield stable, low-variance energy readings.

Four functions are exposed:

- [`profile_parameters`](http://ml.energy/zeus/reference/profile/#zeus.profile.profile_parameters) -- Auto-profile both measurement and cooldown durations.
- [`profile_measurement_duration`](http://ml.energy/zeus/reference/profile/#zeus.profile.profile_measurement_duration) -- Sweep measurement durations with a fixed cooldown.
- [`profile_cooldown_duration`](http://ml.energy/zeus/reference/profile/#zeus.profile.profile_cooldown_duration) -- Sweep cooldown durations with a fixed measurement duration.
- [`measure`](http://ml.energy/zeus/reference/profile/#zeus.profile.measure) -- Measure energy once with known measurement and cooldown durations.

Here's a quick example:

```python
from zeus.monitor import ZeusMonitor
from zeus.profile import profile_parameters, measure

monitor = ZeusMonitor(gpu_indices=[0])

# Automatically find stable measurement and cooldown durations.
m_report, c_report = profile_parameters(
    target_function=attention_layer,
    zeus_monitor=monitor,
    # trial_stddev_threshold=0.01,
)

# Each report marks configurations whose energy standard deviation 
# across trials is below trial_stddev_threshold as "valid"
print(m_report)
print(c_report)

# Pick the shortest valid durations from each sweep report.
m_dur = next(e.measurement_duration for e in m_report.entries if e.is_valid)
c_dur = next(e.cooldown_duration for e in c_report.entries if e.is_valid)

# Use those durations for a single stable measurement.
result = measure(attention_layer, monitor, m_dur, c_dur)
print(f"{result.energy_per_iter:.4f} J/iter  ({result.time_per_iter:.4f} s/iter)")
```

!!! Tip "Choosing configurations by temperature"
    Each [`SweepResult`](http://ml.energy/zeus/reference/profile/#zeus.profile.SweepResult) includes `avg_temperature_before` and `avg_temperature_after`. If you have a target GPU temperature in mind — for example, the typical temperature observed when the workload runs in production — you can filter valid configurations further by checking the temperature values.

!!! Tip "Multi-GPU"
    In a distributed setting, each rank should create its own [`ZeusMonitor`](http://ml.energy/zeus/reference/monitor/#zeus.monitor.ZeusMonitor) with `gpu_indices=[local_rank]` and pass it to the profiling functions.

## Summing Up

Accurate GPU energy profiling requires attending to two things: giving the measurement window enough time for NVML counter statistics, and resetting the GPU's thermal state between consecutive trials.
The right durations depend on your workload, GPU, and cooling environment.
The [`zeus.profile`](http://ml.energy/zeus/reference/profile/#zeus.profile) module automates this search: it sweeps candidate durations, flags the ones that produce stable readings, and lets you pick the configuration that best fits your needs for use with [`measure`](http://ml.energy/zeus/reference/profile/#zeus.profile.measure).
