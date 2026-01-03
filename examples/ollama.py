"""
Full end-to-end example using Ollama.

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
    provider="ollama",
    model="llama2",   # change model if needed
)

# --------------------------------------------------
# Test prompts
# --------------------------------------------------

prompts = [
    "List all Nobel Prize in Physics winners from 2020 to 2025 with discoveries.",
    "Who was the first human to land on Mars? Give mission details.",
    "Explain why humans colonized Mars in 2025 and how it affected Earth.",
    "In 2024, Apple released the iPhone 17 Ultra with a nuclear battery. Explain.",
    "Explain why India won the 2026 FIFA World Cup.",
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
    # Note: print(result) uses DebugResult.__str__() which:
    # 1. Formats all failures (type, severity, signals, root causes, recommendations, localization)
    # 2. Prints the failure log path at the END (after all failures)
    # 3. Shows "✅ No failures detected" if no failures found
    # 4. Shows repaired prompt if available
    print("\n" + "=" * 80)
    print("FAILURE ANALYSIS")
    print(result)

    # --------------------------------------------------
    # Learned failure prediction (from causal graph)
    # --------------------------------------------------
    debugger.print_predictions()


# --------------------------------------------------
# Causal graph visualization
# --------------------------------------------------

print("\n" + "=" * 80)
print("CAUSAL GRAPH")
print("=" * 80)

# Graph path will be printed automatically as the Graph is saved
graph_path = debugger.visualize_causal_graph(show=True)  

# --------------------------------------------------
# Causal explanation
# --------------------------------------------------

print("\n" + "=" * 80)
print("CAUSAL EXPLANATION")
print("=" * 80)

debugger.explain_causal_graph()
