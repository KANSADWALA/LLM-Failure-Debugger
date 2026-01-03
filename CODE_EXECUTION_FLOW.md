## 🔄 Code Execution Flow (Step-by-Step)

This section explains **what happens internally when you call the debugger**, in the exact order the code executes.

---

## 1️⃣ User Calls the Debugger (Public API)

```python
from llm_failure_debugger import Debugger

debugger = Debugger.from_openai(api_key="sk-...")
result = debugger("Who won the Nobel Prize in Physics in 2026?")
```

➡️ This triggers the `Debugger.__call__()` method, which internally calls:

```python
Debugger.debug(prompt, auto_repair=True)
```

---

## 2️⃣ Pre-Output Failure Prediction (Before Generation)

**File:** `core/precheck.py`

```python
precheck_risks = self.precheck.predict(prompt)
```

What happens:

* Detects **risky prompts before model generation**
* Example risks:

  * Future questions → hallucination risk
  * Very short prompts → consistency risk
  * Logic-heavy prompts → reasoning risk

Output example:

```python
{
  FailureType.HALLUCINATION: 0.7,
  FailureType.CONSISTENCY_ERROR: 0.4
}
```

➡️ These risks are stored in `internals.metadata` for later explanation.

---

## 3️⃣ Model Generation via Adapter (Model-Agnostic)

**File:** `adapters/base.py`

```python
output, internals = self.adapter.analyze(prompt)
```

Internally:

```python
output = adapter.generate(prompt)
internals = adapter.extract_internals(prompt, output)
```

Depending on the adapter:

* OpenAI → token logprobs, tool traces
* Ollama / HF → text only
* Custom → user-defined internals

Result:

```python
output = "The Nobel Prize in Physics 2026 was won by John Smith."
internals = ModelInternals(...)
```

---

## 4️⃣ Signal Extraction (Atomic Evidence)

**File:** `analysis/signals.py`

```python
signals = []
for extractor in EXTRACTORS:
    signals.extend(
        extractor.extract(input_text, output_text, internals, capabilities)
    )
```

Each extractor detects **one specific failure signal**:

Examples:

* `TemporalHallucinationExtractor`
* `UngroundedClaimExtractor`
* `EntropySignalExtractor`
* `KnowledgeContradictionExtractor`

Example extracted signals:

```python
[
  FailureSignal(name="temporal_hallucination", value=1.0),
  FailureSignal(name="rag_ungrounded", value=0.6)
]
```

➡️ Signals are **low-level evidence**, not failures yet.

---

## 5️⃣ Failure Inference (Signal → Failure Type)

**File:** `core/inference.py`

```python
failure_hypotheses = self.inference_engine.infer(signals)
```

Internally:

```python
score += signal.value * weight * signal.confidence
confidence = 1 - exp(-score)
```

Output:

```python
[
  (FailureType.HALLUCINATION, 0.88, ["temporal_hallucination", "rag_ungrounded"])
]
```

➡️ Now the system knows **WHAT failed**, with confidence.

---

## 6️⃣ Root Cause Analysis (Why It Failed)

**File:** `analysis/root_cause.py`

```python
root_causes = self.root_cause_analyzer.analyze(relevant_signals)
```

Signal → Root Cause mapping:

```python
"temporal_hallucination" → KNOWLEDGE_CONTRADICTION
"rag_ungrounded"        → UNGROUNDED_GENERATION
```

Output:

```python
[
  RootCause.KNOWLEDGE_CONTRADICTION,
  RootCause.UNGROUNDED_GENERATION
]
```

➡️ This answers **WHY the failure occurred**.

---

## 7️⃣ Failure Localization (Where It Failed)

**File:** `analysis/attention_localization.py`

```python
failure.localization = FailureLocalization(
    level="knowledge_retrieval_stage",
    evidence=["rag_ungrounded"],
    confidence=0.88
)
```

Examples of stages:

* knowledge retrieval
* reasoning
* decoding
* tool planning

➡️ This answers **WHERE in the pipeline it failed**.

---

## 8️⃣ Causal Graph Update (Learning Across Runs)

**File:** `analysis/causal_graph.py`

```python
self._causal_failures.append(failure)
samples = self._causal_builder.build(self._causal_failures)
self._causal_graph = self._causal_learner.learn(samples)
```

Learned structure:

```
rag_ungrounded → ungrounded_generation → hallucination
temporal_hallucination → knowledge_contradiction → hallucination
```

➡️ The system **learns failure patterns over time**.

---

## 9️⃣ Recommendation Generation (What To Fix)

**File:** `analysis/recommendations.py`

```python
recommendations = self.recommendation_engine.recommend(
    root_causes,
    causal_graph=self._causal_graph
)
```

Example output:

```python
[
  "Add retrieval-augmented generation (RAG)",
  "Force model to say 'I don't know' if unsure",
  "Add temporal grounding instructions"
]
```

---

## 🔟 Automated Prompt Repair (Optional)

**File:** `utils/repair.py`

```python
repaired_prompt, status = self.repair_engine.repair(
    original_prompt,
    [best_root_cause]
)
```

Repaired prompt:

```text
Who won the Nobel Prize in Physics in 2026?

Instruction:
- Cite sources and say 'I don't know' if unsure.
- Do not assume the premise is correct.
```

➡️ Only triggered when:

* Severity ≥ MEDIUM
* Confidence ≥ 0.6

---

## 1️⃣1️⃣ Repair Safety Evaluation (Second Pass)

```python
repair_safe = RepairEvaluator.is_safe(repaired_output)
```

Checks for:

* Safe abstention
* Premise rejection
* No hallucinated facts

---

## 1️⃣2️⃣ Logging, Tracking & Active Learning

### Failure Logging

```python
failures_log_YYYY-MM-DD_HH-MM-SS.jsonl
```

### Failure Tracking

**File:** `tracking/tracker.py`

```python
self.tracker.log(model_name, model_version, failures)
```

### Active Learning Selection

**File:** `tracking/active_learning.py`

```python
selected = ActiveLearningSelector().select(failures, k=10)
```

➡️ Selects failures for:

* Human review
* Dataset augmentation
* Retraining

---

## 1️⃣3️⃣ Final DebugResult Returned

```python
DebugResult(
  failures=[FailureInstance(...)],
  signals=[FailureSignal(...)],
  metadata={
    "failure_log_path": "...",
    "repaired_prompt": "...",
    "repair_safe": True
  }
)
```

User-friendly access:

```python
result.has_failures
result.recommendations
result.repaired_prompt
```

---

## ✅ Summary (One-Line Flow)

```
Prompt → Precheck → LLM → Signals → Failure → Root Cause
      → Causal Learning → Repair → Evaluation → Logging
```

---