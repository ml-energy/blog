---
date: 2023-07-24
authors:
  - jaywonchung
categories:
  - energy
  - measurement
links:
  - ZeusMonitor: https://ml.energy/zeus/reference/monitor/#zeus.monitor.ZeusMonitor
---

# Measuring GPU Energy: Best Practices

To optimize something, you need to be able to measure it right.
In this post, we'll look into potential pitfalls and best practices for GPU energy measurement.

<!-- more -->

## Best practices

### 1. Actually measure energy

We sometimes see energy consumption or carbon emission being estimated with back-of-the-envelope calculations using the GPUs' Thermal Design Power (TDP), or in other words their maximum power consumption.[^1]
This is of course fine if you're trying to raise awareness and motivate others to start looking at energy consumption.
However, if we want to really observe what happens to energy when we change around parameters and optimize it, we must actually measure energy consumption.

TDP is usually not the best proxy for GPU power consumption.
Below is the average power consumption of one NVIDIA V100 GPU while training different models:
<figure markdown>
  ![Average power V100](assets/correctly-measuring-energy/avg-power-v100-light.svg#only-light)
  ![Average power V100](assets/correctly-measuring-energy/avg-power-v100-dark.svg#only-dark)
  <figcaption>GPU TDP is not the best proxy for average power consumption.</figcaption>
</figure>
You can see that depending on the computation characteristic and load of models, average power consumption varies significantly, and never really touches the GPU's TDP.

What about for larger models?
Below we measured the average power consumption of four NVIDIA A40 GPUs training larger models with four-stage pipeline parallelism:
<figure markdown>
  ![Average power A40](assets/correctly-measuring-energy/avg-power-a40-light.svg#only-light)
  ![Average power A40](assets/correctly-measuring-energy/avg-power-a40-dark.svg#only-dark)
  <figcaption>GPU TDP is not the best proxy for average power consumption.</figcaption>
</figure>
It's the same again.
Even for computation-heavy large model training, GPU average power consumption does not reach TDP.

!!! Takeaway
    GPU Thermal Design Power (TDP) is not the best estimate.
    Actually measure power and energy.


### 2. Use the most efficient API

How do you measure GPU energy?
Depending on the microarchitecture of your GPU, there can be either one way or two ways.
Below, we show the sample code using [Python bindings for NVML](https://pypi.org/project/nvidia-ml-py/).

=== "Power API (All microarchitectures)"

    ```python
    import pynvml

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # First GPU
    power = pynvml.nvmlDeviceGetPowerUsage(handle)  # milliwatts
    ```

    The power API returns the current power consumption of the GPU.
    Since energy is power integrated over time, you will need a separate thread to poll the power API, and later integrate the power samples over time using [`sklearn.metrics.auc`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html), for example.

=== "Energy API (Volta or newer)"

    ```python
    import pynvml

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # First GPU
    energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)  # millijoules
    ```

    The energy API returns the total energy consumption of the GPU *since the driver was last loaded*.
    Therefore, you just need to call the energy API once before computing and once after computing, and subtract the two for the energy consumption between the two calls.

The power API, although supported by all microarchitectures, requires *polling* and then one discrete integration across time to compute energy.
While polling happens in just one CPU core, it will be kept at high utilization during training and will consume some amount of extra energy purely for energy monitoring.
On the other hand, the energy API simply requires two function calls in the main thread and one subtraction.

!!! Takeaway
    Use `nvmlDeviceGetTotalEnergyConsumption` if your GPU is Volta or newer. Otherwise, you'll need to poll `nvmlDeviceGetPowerUsage` and integrate power measurements across time to obtain energy.


### 3. Synchronize CPU and GPU

In most DNN training frameworks, the CPU dispatches CUDA kernels to the GPU in an asynchronous fashion.
In other words, the CPU tells the GPU to do some work and moves on to the next line of Python code without waiting for the GPU to complete.
Consider this:

```python
import time
import torch

start_time = time.time()
train_one_step()
elapsed_time = time.time() - start_time  # Wrong!
```

The time measured is likely an underestimation of GPU computation time, as the CPU code is running ahead of GPU code and the GPU may not have finished executing all the work ordered by the CPU.
Therefore, there are APIs in PyTorch[^2] and JAX[^3] that *synchronize* CPU and GPU execution, i.e., have the CPU wait for the GPU is done:

```python
import time
import torch

start_time = time.time()
train_one_step()
torch.cuda.synchronize()  # Synchronizes CPU and GPU time.
elapsed_time = time.time() - start_time
```

The same is the case for measuring GPU power or energy; NVML code that runs on your CPU must be synchronized with GPU execution for the measurement to be accurate:

```python
import time
import torch
import pynvml

start_time = time.time()
start_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
train_one_step()
torch.cuda.synchronize()  # Synchronizes CPU and GPU time.
elapsed_time = time.time() - start_time
consumed_energy = \
    pynvml.nvmlDeviceGetTotalEnergyConsumption(handle) - start_energy
```

!!! Takeaway
    To accurately measure GPU time and energy consumption, make the CPU wait for GPU work to complete. For example, in PyTorch, use [`torch.cuda.synchronize`](https://pytorch.org/docs/stable/generated/torch.cuda.synchronize.html).


## The all-in-one solution: [`ZeusMonitor`](https://ml.energy/zeus/reference/monitor/#zeus.monitor.ZeusMonitor)

Do all these feel like a headache?
Well, [`ZeusMonitor`](https://ml.energy/zeus/reference/monitor/#zeus.monitor.ZeusMonitor) got you covered.
Simple.
It implements all three best practices and provides a simple interface:

```python
from zeus.monitor import ZeusMonitor

monitor = ZeusMonitor(gpu_indices=[0, 2])  # Arbitrary GPU indices.
                                           # Respects `CUDA_VISIBLE_DEVICES`.

monitor.begin_window("training")
# Train!
measurement = monitor.end_window("training", sync_cuda=True)

print(f"Time elapsed: {measurement.time}s")
print(f"GPU0 energy consumed: {measurement.energy[0]}J")
print(f"GPU2 energy consumed: {measurement.energy[2]}J")
print(f"Total energy consumed: {measurement.total_energy}J")
```

`ZeusMonitor` will automatically detect your GPU architecture (separately for each GPU index you with to monitor) and use the right NVML API to measure GPU time and energy consumption.
You can have multiple overlapping *measurement windows* as long as you choose different names for them.

Sounds good?
Get started with Zeus [here](https://ml.energy/zeus){.external}!


[^1]:
    For instance,
    ["Estimating the Carbon Footprint of BLOOM, a 176B Parameter Language Model"](https://arxiv.org/pdf/2211.02001v1.pdf) and
    ["Llama 2: Open Foundation and Fine-Tuned Chat Models"](https://arxiv.org/pdf/2307.09288v2.pdf).
[^2]: [`torch.cuda.synchronize`](https://pytorch.org/docs/stable/generated/torch.cuda.synchronize.html)
[^3]: [`jax.block_until_ready`](https://jax.readthedocs.io/en/latest/async_dispatch.html)
