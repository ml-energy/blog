---
date: 2023-07-19
authors:
  - jaywonchung
categories:
  - energy
---

# Accounting GPU energy consumption correctly

To optimize something, you need to be able to measure it correctly.
However, nowadays, I see that many people are not measuring correctly, or sometimes, not even measuring.
Let's look into this.

<!-- more -->

## Bad examples

- [`Llama 2`](https://ml.energy)

```python
from zeus.monitor import ZeusMonitor
```
