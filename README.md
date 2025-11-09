# Baby Dragon Hatchling

## **Bridging the Gap Between Transformers and the Brain**

**Baby Dragon Hatchling (BDH)** is a biologically inspired large language model architecture that connects principles of deep learning with the foundations of neuroscience. Developed by researchers at [Pathway](https://pathway.com), BDH provides a theoretical and practical framework for understanding the emergence of reasoning and generalization in artificial systems.

This repository contains the official implementation from the paper:
> *A. Kosowski, P. Uznański, J. Chorowski, Z. Stamirowska, M. Bartoszkiewicz.*
> [_The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain_](https://doi.org/10.48550/arXiv.2509.26507), arXiv (2025).


## Overview

BDH represents a **scale-free, locally interacting network of neurons** capable of intrinsic reasoning dynamics. BDH scales like a Transformer on performance benchmarks—yet retains full interpretability and theoretical grounding in the fine-grained dynamics of neuron interactions.

**Key properties:**

- **Scale-free network topology** mimicking biological connectivity
- **Locally interacting neuron particles** with excitatory/inhibitory dynamics
- **Hebbian working memory** based on synaptic plasticity, displaying monosemanticity
- **GPU-friendly state-space formulation** for efficient implementation
- **Interpretable activations** that are sparse and positive

BDH formalizes a bridge between **neural computation and machine-based language understanding**. It shows how **macro reasoning behavior** in large AI models emerges from **micro-level neuron dynamics**, guided by principles of graph theory and local computation.

Empirically, BDH matches **GPT-2–scale Transformers** across language and translation tasks at equivalent parameter scales (10M–1B).


***

## Architecture

<img src="figs/architecture.png" width="600"/>

***

## Relation to Transformers

<img src="figs/vocab.png" width="600"/>

BDH and the Transformer share attention-inspired computation; however, BDH’s graph-based architecture makes its attention **emerge naturally from neuron-level interactions**, reflecting attention as seen in biological systems.

***

## Scaling Laws

<img src="figs/bdh_scaling.png" width="600"/>

BDH follows **Transformer-like scaling laws**, maintaining parameter efficiency while achieving interpretability at any scale.

***

## Installation and Training

```bash
# install dependencies
pip install -r requirements.txt

# train BDH on a toy dataset
python train.py
```

### Low-Precision Training (FP8 & 4-bit)

BDH supports advanced low-precision training for memory efficiency and faster training:

#### FP8 Training (Hopper H100 / Blackwell GPUs)

For NVIDIA H100 or Blackwell GPUs with native FP8 tensor cores:

```bash
# install transformer engine
pip install transformer-engine[pytorch]
```

Enable in `train.py`:
```python
USE_FP8 = True  # Up to 2x faster training, minimal accuracy loss
```

**Benefits:** Hardware-accelerated FP8, up to 2x faster than BF16, best for H100/Blackwell

#### 4-bit Optimizer Training (Any CUDA GPU)

For memory-efficient training on any NVIDIA GPU:

```bash
# install bitsandbytes
pip install bitsandbytes
```

Enable in `train.py`:
```python
USE_4BIT_OPTIMIZER = True  # Reduces memory by ~50-75%
```

**Benefits:** Works on any CUDA GPU (RTX 3090, 4090, A100), 50-75% memory reduction

See [4BIT_TRAINING.md](4BIT_TRAINING.md) for detailed configuration and GPU compatibility.

## Inference and Prompting

After training, you can load a checkpoint and prompt the model:

```bash
# Interactive mode - prompt the model interactively
python prompt.py checkpoints/checkpoint_1000.pt

# Single prompt mode
python prompt.py checkpoints/checkpoint_1000.pt --prompt "The Capital of France is"

# Customize generation parameters
python prompt.py checkpoints/checkpoint_1000.pt --prompt "Once upon a time" --max-tokens 200 --top-k 5 --temperature 0.8
```

**Interactive mode commands:**
- Type any text to generate a completion
- Type `settings` to adjust generation parameters (max tokens, top-k, temperature)
- Type `quit` or `exit` to stop

<!--For visualization and interpretability analysis, explore the example notebooks in `notebooks/`.-->



## Learn and Discuss

- Watch the *SuperDataScience podcast* [▶️ *Dragon Hatchling: The Missing Link Between Transformers and the Brain*](https://www.youtube.com/watch?v=mfV44-mtg7c) (72 min.) featuring Adrian Kosowski in conversation with Jon Krohn, unpacking BDH’s neuron-level architecture and sparse reasoning dynamics.

- Read about BDH in
[*Forbes*](https://www.forbes.com/sites/victordey/2025/10/08/can-ai-learn-and-evolve-like-a-brain-pathways-bold-research-thinks-so/),
[*Semafor*](https://www.semafor.com/article/10/01/2025/new-ai-research-claims-to-be-getting-closer-to-modeling-human-brain),
[*The Turing Post*](https://www.turingpost.com/p/fod-121-300-million-to-start-a-big-promise-for-science#the-freshest-research-papers-catego),
[*Quantum Zeitgeist*](https://quantumzeitgeist.com/palo-alto-ai-firm-pathway-unveils-post-transformer-architecture-for-autonomous-ai/),
[*Golem*](https://www.golem.de/news/neue-ki-architektur-was-ist-baby-dragon-hatchling-2510-201047-2.html),
and elsewhere in the media.

- Discuss and share the BDH paper on:
[*Hugging Face Papers*](https://huggingface.co/papers/2509.26507), 
[*Alphaxiv*](https://alphaxiv.org/abs/2509.26507),
and [*EmergentMind*](https://emergentmind.com/papers/2509.26507).

## Community Projects

- [adamskrodzki/bdh](https://github.com/adamskrodzki/bdh): dynamic vocabulary, stateful attention
- [mosure/burn_dragon_hatchling](https://github.com/mosure/burn_dragon_hatchling): Burn port
- [severian42/bdh](https://github.com/severian42/bdh): MLX port
- [Git-Faisal/bdh](https://github.com/Git-Faisal/bdh)
- [GrahLnn/bdh](https://github.com/GrahLnn/bdh)

## Acknowledgements
We thank Andrej Karpathy for the [nanoGPT](https://github.com/karpathy/nanoGPT/) code and the tiny Shapespeare dataset used in this demonstration.

BDH research stands at the intersection of **AI architecture**, **biological learning models**, and **theoretical computer science**—an effort to map the *equations of reasoning* between artificial and biological intelligence.
