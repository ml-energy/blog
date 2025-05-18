---
date: 2025-05-17
authors:
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

The main tool available is a built-in macOS CLI utility called `powermetrics`, which prints energy metrics to output at set time intervals. However, it's annoying to use this *programmatically* (e.g., for precisely teasing apart energy consumption by different parts of code), since it measures and reports energy over set time intervals (e.g., 500 ms) instead of at arbitrary start/end points in your code. You would also need to parse the output of the tool yourself in a background thread or process.

So, to make this easier, we built [zeus-apple-silicon](https://github.com/ml-energy/zeus-apple-silicon), a very small library designed specifically for energy profiling on Apple Silicon.

Our hope with this library was to make something more straightforward to use -- it allows you to measure energy over any arbitrary block of code without needing periodic parsing in background threads. As a bonus, it provides more detailed readings than powermetrics. Whereas powermetrics only provides aggregate results for CPU, GPU, and ANE, this library gives you per-core energy (each efficiency/performance core separately), DRAM energy, and so on.

The library is written/available in C++, but it’s importable as a Python package via bindings. So, if you just want to know how much energy it takes to prompt a model, it can be as simple as:

```python
# Install via `pip install zeus-apple-silicon`
from zeus_apple_silicon import AppleEnergyMonitor
from llama_cpp import Llama

# (1) Initialize your model
llm = Llama(
    model_path="./models/llama-2-7b-chat.Q4_K_M.gguf",
    verbose=False,
)

# (2) Initialize energy monitor
monitor = AppleEnergyMonitor()

# (3) See how much energy is consumed while generating response
monitor.begin_window("prompt") # START an energy measurement window
output = llm(
    "Q: Who are you? A: ",
    max_tokens=32,
    stop=["Q:", "\n"],
)
energy_metrics = monitor.end_window("prompt") # END measurement, get results

text_answer = output["choices"][0]["text"]
print(text_answer)

# (4) Print energy usage over the measured window
print("--- Prompt Energy ---")
print(energy_metrics)
```

And you might see output like:

```
CPU Total: 934 mJ
Efficiency cores: 34 mJ  34 mJ  21 mJ  14 mJ  
Performance cores: 308 mJ  118 mJ  88 mJ  54 mJ  
Efficiency core manager: 72 mJ
Performance core manager: 191 mJ
DRAM: None (unavailable)
GPU: 20708 mJ
GPU SRAM: None (unavailable)
ANE: None (unavailable)
```

*Note: above, some fields are marked as `None (unavailable)`, which happens when a processor doesn't support energy metrics for that field. On M1 chips (like on my own machine), DRAM, ANE, and GPU SRAM results are often unavailable. On newer machines (M2 and above), all fields are typically present.*

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

In terms of granularity, energy readings are updated basically as fast as the processor’s energy counters are updated and passed through IOKit APIs, which is what the tool uses internally. When I tested this locally, updates were happening at less than 1 millisecond granularity.

This library works as a standalone tool, but it was developed as part of a larger project at the University of Michigan called [Zeus](https://ml.energy/zeus/) (github: [https://github.com/ml-energy/zeus](https://github.com/ml-energy/zeus)), aimed at measuring/optimizing deep learning energy usage, particularly on GPUs. It supports NVIDIA GPUs and AMD GPUs (and Intel CPUs and some other SoCs as well) and offers optimization capabilities alongside measurement.
