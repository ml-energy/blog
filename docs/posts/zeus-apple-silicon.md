---
date: 2025-05-17
authors:
  - michahn01
  - jaywonchung
categories:
  - energy
  - measurement
links:
  - ZeusMonitor: https://ml.energy/zeus/reference/monitor/#zeus.monitor.ZeusMonitor
  - zeus-apple-silicon: https://github.com/ml-energy/zeus-apple-silicon
---

# Profiling LLM energy consumption on Macs

If you want to see how much energy LLM inference consumes on Apple Silicon, it's hard to find a straightforward way to do this programmatically, from within code. In this post, we'll explore how we can do this.

<!-- more -->

The main tool available is a built-in macOS CLI utility called `powermetrics`, which prints energy metrics to output at set time intervals. However, it's annoying to use this *programmatically* (e.g., for precisely teasing apart energy consumption by different parts of code), because:

1. It requires `sudo`
1. It measures and reports energy over set time intervals (e.g., 500 ms) instead of at arbitrary start/end points in your code
1. You have to parse the output of the tool yourself in a background thread or process.

So, to make this easier, we built [zeus-apple-silicon](https://github.com/ml-energy/zeus-apple-silicon), a very small C++/Python library designed specifically for energy profiling on Apple Silicon.

Our hope with this library was to make something more straightforward to use -- it allows you to measure energy over any arbitrary block of code without needing periodic parsing in background threads. As a bonus, it provides more detailed readings than `powermetrics`. Whereas `powermetrics` only provides aggregate results for CPU, GPU, and ANE, this library gives you per-core energy (each efficiency/performance core separately), DRAM energy, and so on.

The library is written/available in C++, but it’s importable as a Python package via bindings.
It can be installed with:

```bash
pip install zeus-apple-silicon
```

So, if you just want to know how much energy it takes to prompt a model, it can be as simple as:

```python
# Assumes `pip install llama-cpp-python huggingface-hub`

from zeus_apple_silicon import AppleEnergyMonitor
from llama_cpp import Llama

# (1) Initialize your model
llm = Llama.from_pretrained(
    repo_id="bartowski/Llama-3.2-3B-Instruct-GGUF",
    filename="Llama-3.2-3B-Instruct-Q6_K.gguf",
    n_gpu_layers=-1,
)

# (2) Initialize energy monitor
monitor = AppleEnergyMonitor()

# (3) See how much energy is consumed while generating response
monitor.begin_window("prompt") # START an energy measurement window
output = llm.create_chat_completion(
      messages = [
          {"role": "system", "content": "You are a helpful assistant."},
          {"role": "user", "content": "What makes a good Python library? Answer concisely."}
      ],
)
energy_metrics = monitor.end_window("prompt") # END measurement, get results

print("--- Model Output ---")
print(output["choices"][0]["message"]["content"])

# (4) Print energy usage over the measured window
print("--- Energy ---")
print(energy_metrics)
```

And the output might look something like this:

```
--- Energy ---
CPU Total: 25602 mJ
Efficiency cores: 267 mJ  245 mJ
Performance cores: 5200 mJ  5268 mJ  4205 mJ  2678 mJ  1723 mJ  538 mJ  739 mJ  332 mJ
Efficiency core manager: 301 mJ
Performance core manager: 4104 mJ
DRAM: 16347 mJ
GPU: 81962 mJ
GPU SRAM: 4 mJ
ANE: 0 mJ
```

!!! Note
    Some fields may be `None`, which happens when a processor doesn't support energy metrics for that field. On M1 chips, for instance, DRAM, ANE, and GPU SRAM results may not be available. On newer machines (M2 and above), all fields are typically present.

Alternatively, if you’re interfacing with low-level inference code directly (say, in llama.cpp), you can use the C++ version of the energy profiler, which is available as a header-only include, to tease apart energy metrics more precisely.

For example, in a typical llama.cpp inference setup, you might repeatedly call `llama_decode` to run your model’s forward pass over a batch of one or more tokens. So, you can wrap the `llama_decode` call in an energy profiling window, like this:

```cpp
#include <apple_energy.hpp>

/* ... Load model, initialize context, etc. */

AppleEnergyMonitor monitor;

while ( /* inference not finished */ ) {

    // START energy measurement here
    monitor.begin_window("batch");

    llama_decode(context, batch);

    // END energy measurement here
    AppleEnergyMetrics metrics = monitor.end_window("batch");

    // `metrics` contains energy consumed during call to `llama_decode`
}
```

And this measurement would give you insight into the energy consumption per batch or token, ignoring one-time costs like model loading or context initialization.

!!! Tip
    You can obtain `apple_energy.hpp` from the [zeus-apple-silicon GitHub repository](https://github.com/ml-energy/zeus-apple-silicon).

In terms of granularity, energy readings are updated basically as fast as the processor’s energy counters are updated and passed through IOKit APIs, which is what the tool uses internally. When tested locally, updates were happening at less than 1 millisecond granularity.

This library works as a standalone tool, but it was developed as part of a larger project called [Zeus](https://ml.energy/zeus/) (GitHub: [https://github.com/ml-energy/zeus](https://github.com/ml-energy/zeus)), aimed at measuring/optimizing deep learning energy usage, particularly on GPUs. It offers the same window-based measurement API like `zeus-apple-silicon`, but supports broader hardware like NVIDIA and AMD GPUs, CPUs (mostly), DRAM (mostly), Apple Silicon, and NVIDIA Jetson platforms, and offers automated energy optimizers for Deep Learning scenarios alongside measurement.
