---
{{ card_data }}
---

# Model Card for REGENT: A Retrieval-Augmented Generalist Agent

### Model Sources

- **Repository:** https://github.com/regent-research/regent

## How to Get Started with the Model

Use the code below to get started with the model.

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("{{ model_name | default("[More Information Needed]", true)}}")
```

