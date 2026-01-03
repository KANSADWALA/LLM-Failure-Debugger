"""
Full end-to-end example using Hugging Face local models.

This file demonstrates ALL core capabilities of llm_failure_debugger:
- LLM usage
- failure detection
- root cause analysis
- recommendations
- explicit failure localization
- learned failure prediction
- causal graph learning
- causal graph visualization
- causal explanation
"""

import sys
from pathlib import Path

# Add parent directory to path so we can import llm_failure_debugger
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_failure_debugger import Debugger

# --------------------------------------------------
# Create debugger (ONLY public API) 
# --------------------------------------------------

debugger = Debugger.from_llm(
    provider="huggingface",
    model="gpt2",  # or mistralai/Mistral-7B-Instruct
)

# --------------------------------------------------
# Test prompts
# --------------------------------------------------

prompts = [
    "Who was the first human to land on Mars?",
    "Explain why India won the 2026 FIFA World Cup.",
    "List all Nobel Prize winners in Physics from 2020 to 2025.",
    "In 2024 Apple released an iPhone with a nuclear battery. Explain.",
]


# --------------------------------------------------
# Run prompts
# --------------------------------------------------

for i, prompt in enumerate(prompts, 1):
    print("\n" + "=" * 80)
    print(f"PROMPT {i}")
    print("=" * 80)
    print(prompt)

    # ------------------------------
    # Run model through debugger
    # ------------------------------
    result = debugger(prompt)

    # ------------------------------
    # Model output
    # ------------------------------
    print("\nMODEL OUTPUT")
    print("-" * 80)
    print(result.output_text)

    # ------------------------------
    # Failure analysis
    # ------------------------------
    if not result.has_failures:
        print("\n✅ No failures detected.")
        continue

    print("\n🚨 FAILURES DETECTED")

    for failure in result.failures:
        print("\n------------------------------")
        print(f"Failure Type : {failure.failure_type.value}")
        print(f"Severity     : {failure.severity.value}")
        print(f"Confidence   : {failure.confidence:.2f}")

        if failure.signals:
            print("Signals:")
            for s in failure.signals:
                print(f"  - {s}")

        if failure.root_causes:
            print("Root Causes:")
            for rc in failure.root_causes:
                print(f"  - {rc}")

        if failure.recommendations:
            print("Recommendations:")
            for rec in failure.recommendations:
                print(f"  - {rec}")

        # ------------------------------
        # Explicit failure localization
        # ------------------------------
        if failure.localization:
            print("Localization:")
            print(f"  Level      : {failure.localization.level}")
            print(f"  Evidence   : {failure.localization.evidence}")
            print(f"  Confidence : {failure.localization.confidence:.2f}")

    # ------------------------------
    # Prompt repair (automatic)
    # ------------------------------
    if result.repaired_prompt:
        print("\n🔧 REPAIRED PROMPT SUGGESTED")
        print("-" * 80)
        print(result.repaired_prompt)


# --------------------------------------------------
# Learned failure prediction (from causal graph)
# --------------------------------------------------

print("\n" + "=" * 80)
print("FAILURE PREDICTION (LEARNED)")
print("=" * 80)

predictions = debugger.predict_failures()

if not predictions:
    print("No learned failure patterns yet.")
else:
    for p in predictions:
        print(
            f"Likely failure: {p.failure_type} | "
            f"Probability: {p.probability:.2f} | "
            f"Based on: {p.based_on}"
        )


# --------------------------------------------------
# Causal graph visualization
# --------------------------------------------------

print("\n" + "=" * 80)
print("CAUSAL GRAPH")
print("=" * 80)

graph_path = debugger.visualize_causal_graph(show=True)


if graph_path:
    print(f"📈 Causal graph saved to: {graph_path}")


# --------------------------------------------------
# Causal explanation
# --------------------------------------------------

print("\n" + "=" * 80)
print("CAUSAL EXPLANATION")
print("=" * 80)

debugger.explain_causal_graph()
