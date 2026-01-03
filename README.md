# LLM Failure Debugger

A **production-grade, model-agnostic framework** for **diagnosing, explaining, and mitigating Large Language Model (LLM) failures** using structured signals, causal reasoning, and automated prompt repair.

This library goes beyond accuracy metrics to answer:

> **Why did the model fail?
> Where did it fail in the reasoning pipeline?
> What is the most effective fix?**

---

# 📌 Problem Statement

Large Language Models (LLMs) exhibit increasingly strong performance across reasoning, generation, and tool-augmented tasks; however, they remain prone to systematic failures such as hallucinations, logical inconsistencies, temporal errors, and tool misuse. Existing evaluation methodologies primarily focus on output correctness, benchmark accuracy, or post-hoc explainability, offering limited insight into why a failure occurred, where it originated in the model’s reasoning pipeline, and how it can be reliably mitigated.

Current LLM debugging practices largely treat failures as opaque outcomes, rather than as diagnosable system-level events. As a result:

<ul>
<li>Failures are detected after generation, with minimal internal attribution.</li>
<li>Root causes are often conflated (e.g., hallucination vs. knowledge contradiction).</li>
<li>Debugging actions rely on ad-hoc prompt tuning or retraining, without structured diagnosis.</li>
<li>There is no unified framework connecting failure signals, internal causes, localization, and actionable repair.</li>
</ul>

This research addresses the absence of a failure-aware debugging framework for LLMs that treats failures as first-class entities, enabling systematic diagnosis rather than superficial correction.

How can we systematically identify, localize, explain, and mitigate internal failure modes of Large Language Models—across reasoning, grounding, and generation—using a structured, model-agnostic debugging framework that links observable failure signals to root causes and actionable repair strategies?

## 🔍 Key Capabilities

* Model-agnostic LLM debugging (OpenAI, Anthropic, Ollama, HuggingFace, Custom)
* Failure signal extraction (hallucination, reasoning breakdown, inconsistency, tool misuse)
* Root-cause analysis with explainable mappings
* Online causal graph learning across runs
* Automated prompt repair with safety evaluation
* Benchmarking & evaluation metrics
* Failure tracking, active learning, and training intervention planning

---

## 🧠 System Architecture

```
┌──────────────┐
│   User Prompt│
└──────┬───────┘
       ↓
┌────────────────────┐
│ Pre-Output Predictor│
│ (risk estimation)   │
└──────┬─────────────┘
       ↓
┌────────────────────┐
│  LLM Adapter Layer │  ← OpenAI / Anthropic / Ollama / HF / Custom
└──────┬─────────────┘
       ↓
┌────────────────────┐
│ Signal Extraction  │  ← entropy, grounding, logic, tools, time
└──────┬─────────────┘
       ↓
┌────────────────────┐
│ Failure Inference  │  ← weighted causal rules
└──────┬─────────────┘
       ↓
┌────────────────────┐
│ Root Cause Analysis│
└──────┬─────────────┘
       ↓
┌────────────────────┐
│ Causal Graph Model │  ← learned across runs
└──────┬─────────────┘
       ↓
┌────────────────────┐
│ Prompt Repair Engine│
└──────┬─────────────┘
       ↓
┌────────────────────┐
│ Evaluation & Logs  │
└────────────────────┘
```

---

## 📁 Directory Structure

```
llm_failure_debugger/
│
├── adapters/                 # Model integration layer
│   ├── base.py               # Abstract adapter interface
│   ├── factory.py            # Adapter factory
│   ├── openai_adapter.py
│   ├── anthropic_adapter.py
│   ├── ollama_adapter.py
│   ├── huggingface_adapter.py
│   └── custom_adapter.py
│
├── core/                     # Core debugging logic
│   ├── debugger.py           # Public API entry point
│   ├── inference.py          # Failure inference engine
│   └── precheck.py           # Pre-generation risk prediction
│
├── analysis/                 # Explanation & causality
│   ├── signals.py            # Failure signal extractors
│   ├── root_cause.py         # Signal → root cause mapping
│   ├── attention_localization.py
│   ├── causal_graph.py       # Causal graph learning & visualization
│   ├── causal_model.py       # Structural causal model
│   └── recommendations.py    # Fix suggestions
│
├── utils/
│   └── repair.py             # Prompt repair & safety evaluation
│
├── evaluation/
│   ├── benchmark.py          # Benchmark runner
│   └── metrics.py            # Precision / Recall / F1
│
├── tracking/
│   ├── tracker.py            # Failure tracking over time
│   └── active_learning.py    # Sample selection for retraining
│
├── training/
│   └── intervention.py       # Training intervention planner
│
├── type_definitions.py       # Shared enums & dataclasses
├── __init__.py               # Public exports
├── pyproject.toml            # Packaging & dependencies
└── README.md
```

---

## 🔄 Code Execution Flow

### 1. Debugger Initialization

```python
from llm_failure_debugger import Debugger

debugger = Debugger.from_openai(api_key="sk-...")
```

---

### 2. Prompt Debugging

```python
result = debugger("Who won the Nobel Prize in Physics in 2026?")
```

Internally:

1. Pre-output risk prediction
2. Model generation via adapter
3. Signal extraction
4. Failure inference
5. Root-cause attribution
6. Causal graph update
7. Prompt repair (optional)
8. Failure logging & tracking

---

### 3. Failure Output

```python
if result.has_failures:
    print(result.recommendations)
    print(result.repaired_prompt)
```

---

### 4. Causal Graph Visualization

```python
debugger.visualize_causal_graph()
debugger.explain_causal_graph()
```

---

### 5. Benchmark Evaluation

```python
from llm_failure_debugger import BenchmarkRunner

runner = BenchmarkRunner(debugger)
results = runner.run_from_file("benchmark.json")
print(results["metrics"])
```

---

## 🧪 Failure Types Detected

* Hallucination
* Reasoning Breakdown
* Consistency Error
* Tool Hallucination / Execution Error
* Temporal Hallucination
* Knowledge Contradiction
* Semantic Drift

---

## 🛠 Tech Stack

**Core Language**

* Python 3.10+

**LLM APIs**

* OpenAI API
* Anthropic Claude
* Ollama (local)
* HuggingFace Transformers

**ML / Reasoning**

* Causal inference (SCM-inspired)
* Rule-based probabilistic inference
* Entropy-based uncertainty estimation
* Active learning strategies

**Visualization**

* NetworkX
* Matplotlib

**Evaluation**

* Precision / Recall / F1
* Root-cause attribution F1

**Packaging**

* setuptools
* pyproject.toml

---

## 🎯 Design Philosophy

* **Explainability first**, not accuracy chasing
* **Model-agnostic by design**
* **Safe failure handling via abstention**
* **Causal reasoning over black-box heuristics**
* **Library-first, research-ready architecture**

---

## 📌 Typical Use Cases

* LLM hallucination debugging
* Safety & reliability evaluation
* Research on LLM failure modes
* Prompt robustness testing
* Dataset curation & retraining
* Model comparison across providers

---

## 🚀 Status

**v0.1.0 – Stable research & production foundation**

---
