# llm_failure_debugger/analysis/causal_graph.py

from fileinput import filename
from typing import List, Dict
from dataclasses import dataclass
from collections import defaultdict
import matplotlib.pyplot as plt
import networkx as nx
import itertools
from ..type_definitions import FailureInstance, RootCause

@dataclass
class CausalSample:
    """
    One causal observation from a debugger run.
    """
    signals: Dict[str, int]
    root_causes: Dict[str, int]
    failures: Dict[str, int]


class CausalDatasetBuilder:
    """
    Converts FailureInstances into causal samples.
    """
    def build(self, failures: List[FailureInstance]) -> List[CausalSample]:
        samples = []

        for f in failures:
            signals = {s: 1 for s in f.signals}
            root_causes = {c: 1 for c in f.root_causes}
            failures_map = {f.failure_type.value: 1}

            samples.append(
                CausalSample(
                    signals=signals,
                    root_causes=root_causes,
                    failures=failures_map,
                )
            )
        # --- Counterfactual baseline ---
        samples.append(
            CausalSample(
                signals={},
                root_causes={},
                failures={},
            )
        )

        return samples

class CausalGraph:
    """
    Directed causal graph with edge strengths.
    """
    def __init__(self):
        self.edges: Dict[str, Dict[str, float]] = defaultdict(dict)

    def is_empty(self) -> bool:
        """
        Returns True if the graph has no meaningful causal edges.
        """
        return not any(self.edges.values())

    def is_usable(self) -> bool:
        """
        Returns True if the graph exists and has at least one causal edge.
        """
        return not self.is_empty()

    def add_edge(self, cause: str, effect: str, strength: float):
        self.edges[cause][effect] = strength

    def to_dict(self) -> Dict[str, Dict[str, float]]:
        return self.edges

    def has_node(self, node: str) -> bool:
        return (
            node in self.edges
            or any(node in effects for effects in self.edges.values())
        )

    def out_degree(self, node: str) -> int:
        return len(self.edges.get(node, {}))


class CausalGraphLearner:
    """
    Learns a causal graph from failure data.
    """

    SIGNAL_TIER = 0
    ROOT_CAUSE_TIER = 1
    FAILURE_TIER = 2

    def learn(self, samples: List[CausalSample]) -> CausalGraph:
        graph = CausalGraph()

        variables = self._collect_variables(samples)
        pairs = list(itertools.permutations(variables, 2))
        seen_edges = set()

        for a, b in pairs:
            # --- FIX: enforce causal ordering ---
            tier_a = self._variable_tier(a, samples)
            tier_b = self._variable_tier(b, samples)

            # Enforce strict causal direction:
            # signal -> root_cause -> failure
            if tier_a >= tier_b:
                continue

            
            # Prevent duplicate edges
            edge_key = (a, b)
            if edge_key in seen_edges:
                continue

            strength = self._causal_strength(a, b, samples)
            if strength > 0:
                graph.add_edge(a, b, strength)
                seen_edges.add(edge_key)

        return graph

    def _causal_strength(self, cause: str, effect: str, 
                         samples: List[CausalSample]) -> float:
        joint = 0
        cause_count = 0
        effect_count = 0
        total = len(samples)

        for s in samples:
            has_cause = (
                s.signals.get(cause, 0)
                or s.root_causes.get(cause, 0)
                or s.failures.get(cause, 0)
            )
            has_effect = (
                s.signals.get(effect, 0)
                or s.root_causes.get(effect, 0)
                or s.failures.get(effect, 0)
            )

            if has_cause:
                cause_count += 1
                if has_effect:
                    joint += 1

            if has_effect:
                effect_count += 1

        if cause_count == 0 or total < 2:
            return 0.0

        p_effect_given_cause = joint / cause_count
        p_effect = effect_count / total

        return p_effect_given_cause - p_effect



    def _variable_tier(self, var: str, samples: List[CausalSample]) -> int:
        for s in samples:
            if var in s.signals:
                return self.SIGNAL_TIER
            if var in s.root_causes:
                return self.ROOT_CAUSE_TIER
            if var in s.failures:
                return self.FAILURE_TIER
        return self.FAILURE_TIER

    def _collect_variables(self, samples) -> set[str]:
        vars = set()
        for s in samples:
            vars |= s.signals.keys()
            vars |= s.root_causes.keys()
            vars |= s.failures.keys()
        return vars

    def _causal_dependency(
        self, cause: str, effect: str, samples: List[CausalSample]
        ) -> bool:
        """
        Tests whether P(effect | cause) > P(effect)
        """
        joint = 0
        cause_count = 0
        effect_count = 0
        total = len(samples)

        for s in samples:
            has_cause = (
                s.signals.get(cause, 0)
                or s.root_causes.get(cause, 0)
                or s.failures.get(cause, 0)
            )
            has_effect = (
                s.signals.get(effect, 0)
                or s.root_causes.get(effect, 0)
                or s.failures.get(effect, 0)
            )

            if has_cause:
                cause_count += 1
                if has_effect:
                    joint += 1

            if has_effect:
                effect_count += 1

        # Require at least 2 samples to infer causality
        if total < 2:
            return False

        if cause_count == 0:
            return False

        p_effect_given_cause = joint / cause_count
        p_effect = effect_count / total

        return p_effect_given_cause > p_effect

class CounterfactualAnalyzer:
    """
    Performs simple counterfactual reasoning over the causal graph.
    """

    def estimate_effect(self, graph: CausalGraph, intervention: str) -> Dict[str, float]:
        """
        Estimates downstream effects of removing a variable.
        """
        effects = {}
        
        # Get direct descendants
        for effect in graph.edges.get(intervention, []):
            effects[effect] = 1.0
        
        # ALSO get all transitive descendants (recursive)
        visited = set()
        stack = list(graph.edges.get(intervention, []))
        
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            effects[current] = 1.0
            
            # Add children to stack
            for child in graph.edges.get(current, []):
                if child not in visited:
                    stack.append(child)
        
        return effects
        

class CausalRepairSelector:
    """
    Selects the most impactful repair using the learned causal graph.
    """

    def select_best(
        self,
        root_causes: List[RootCause],
        graph: CausalGraph,
    ) -> RootCause:
        """
        Returns the root cause whose repair removes the most downstream failures.
        """
        scores: Dict[str, int] = {}

        for rc in root_causes:
            scores[rc.value] = self._downstream_impact(rc.value, graph)

        # Select root cause with maximum impact
        best = max(scores.items(), key=lambda x: x[1])[0]
        return RootCause(best)

    def _downstream_impact(self, node: str, graph: CausalGraph) -> int:
        """
        Counts downstream nodes reachable from this root cause.
        """
        visited = set()
        stack = [node]

        while stack:
            current = stack.pop()
            for nxt in graph.edges.get(current, []):
                if nxt not in visited:
                    visited.add(nxt)
                    stack.append(nxt)

        return len(visited)

class CausalGraphVisualizer:
    """
    Matplotlib-based causal graph visualization (no Graphviz).
    """

    def explain_all_edges(self, graph: CausalGraph, samples):
        """
        Dynamically explains all causal edges if possible.
        """
        if not self.validate_graph(graph):
            print(
                "\n[INFO] Skipping causal edge explanations "
                "because the causal graph is not usable."
            )
            return

        if not samples or len(samples) < 2:
            print(
                "\n[INFO] Skipping causal edge explanations "
                "due to insufficient causal samples."
            )
            return

        print("\n==============================")
        print("CAUSAL EDGE EXPLANATIONS")
        print("==============================")

        printed = set()  

        for cause, effects in graph.edges.items():
            for effect in effects:

                # Prevent duplicate explanations
                key = (cause, effect)
                if key in printed:
                    continue          # ✅ PREVENT DUPLICATES
                printed.add(key)

                self.explain_causal_edge(
                    cause=cause,
                    effect=effect,
                    samples=samples,
                )

    def validate_graph(self, graph: CausalGraph) -> bool:
        """
        Checks whether a causal graph is usable before rendering or explaining.
        """
        if graph is None:
            print(
                "\n⚠️ Causal graph could not be learned.\n"
                "Reason: Not enough failure data collected yet.\n"
                "Tip: Run more prompts that trigger failures."
            )
            return False

        if graph.is_empty():
            print(
                "\n⚠️ Causal graph is empty.\n"
                "Reason: No statistically meaningful causal relationships "
                "could be inferred from the data.\n"
                "Tip: More diverse failure patterns are needed."
            )
            return False

        return True

    def render(self, graph: CausalGraph, filename: str = "causal_graph.png",
        show: bool = False) -> str:
        """
        Renders and saves the causal graph as an image.

        Args:
            graph: Learned causal graph
            filename: Output image filename
            show: Whether to display the plot

        Returns:
            Saved image path
        """
        # Validate graph
        if not self.validate_graph(graph):
            return None

        # Create networkx graph
        G = nx.DiGraph()

        # Add edges with weights
        for cause, effects in graph.edges.items():
            for effect, strength in effects.items():
                G.add_edge(cause, effect, weight=strength)

        # --- Layout ---
        # --- Hierarchical layout (by tier) ---
        layers = {0: [], 1: [], 2: []}
        for node in G.nodes:
            layers[self._infer_tier(node)].append(node)

        pos = {}
        x_gap = 2.5
        y_gap = -2.5

        for tier, nodes in layers.items():
            for i, node in enumerate(nodes):
                pos[node] = (i * x_gap, tier * y_gap)

        # --- Node coloring by tier ---
        tier_colors = {
            0: "#90dbf4",  # signal (blue)
            1: "#ffd166",  # root cause (yellow)
            2: "#ef476f",  # failure (red)
        }

        node_colors = [
            tier_colors[self._infer_tier(node)]
            for node in G.nodes
        ]

        # --- Edge thickness by strength ---
        edge_widths = [
            max(1.0, 1.0 + 5.0 * G[u][v]["weight"])
            for u, v in G.edges
        ]

        # --- Plotting ---
        plt.figure(figsize=(14, 10)) # (10,8)
        plt.margins(x=0.25, y=0.25)


        # Set figure title
        fig = plt.gcf()
        try:
            fig.canvas.manager.set_window_title("LLM Failure Causal Graph")
        except Exception:
            pass

        # Draw nodes, edges, labels
        nx.draw_networkx_nodes(
            G,
            pos,
            node_color=node_colors,
            node_size=3000,
            alpha=0.9,
        )

        # Node labels
        nx.draw_networkx_labels(
            G,
            pos,
            font_size=10,
            font_weight="bold",
        )

        # Edges with arrows
        nx.draw_networkx_edges(
            G,
            pos,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=40,
            width=edge_widths,
            edge_color="black",
            alpha=0.9,
            connectionstyle="arc3,rad=0.05",
            min_source_margin=25,
            min_target_margin=30,
        )

        # Edge labels = causal strength
        edge_labels = {
            (u, v): f"+{G[u][v]['weight'] * 100:.0f}%"
            for u, v in G.edges
        }

        # Edge labels
        nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_labels=edge_labels,
            font_size=15,
        )

        plt.axis("off")
        plt.tight_layout(pad=3.0) # 2.0
        plt.savefig(filename, bbox_inches="tight", dpi=200)

        if show:
            print(
                "\nCausal graph opened in a separate window.\n"
                "👉 Close the graph window to continue execution.\n"
                "👉 After closing, detailed causal edge explanations will be shown "
                "in the terminal / PowerShell if you have included "
                "`explain_all_edges()` in your code.\n"
            )

            plt.show()
        else:
            plt.close()

        return filename

    def _infer_tier(self, node: str) -> int:
        if "hallucination" in node or "breakdown" in node:
            return 2  # failure
        if "inconsistency" in node or "drift" in node or "contradiction" in node:
            return 1  # root cause
        return 0      # signal

    @staticmethod
    def explain_causal_edge(cause: str, effect: str, samples):
        """
        Prints a human-readable explanation of a causal edge:
        P(effect | cause) − P(effect)

        Gracefully handles cases where explanation is not possible.
        """
        total = len(samples)

        if total < 2:
            print(
                f"\n[INFO] Not enough data to explain causal edge "
                f"'{cause} → {effect}' (need ≥2 samples)."
            )
            return

        cause_count = 0
        effect_count = 0
        joint_count = 0

        for s in samples:
            has_cause = (
                s.signals.get(cause, 0)
                or s.root_causes.get(cause, 0)
                or s.failures.get(cause, 0)
            )
            has_effect = (
                s.signals.get(effect, 0)
                or s.root_causes.get(effect, 0)
                or s.failures.get(effect, 0)
            )

            if has_effect:
                effect_count += 1

            if has_cause:
                cause_count += 1
                if has_effect:
                    joint_count += 1

        # --- Guard conditions ---
        if cause_count == 0:
            print(
                f"\n[INFO] Cannot explain '{cause} → {effect}': "
                f"cause never appeared in the data."
            )
            return

        if effect_count == 0:
            print(
                f"\n[INFO] Cannot explain '{cause} → {effect}': "
                f"effect never appeared in the data."
            )
            return

        p_effect = effect_count / total
        p_effect_given_cause = joint_count / cause_count
        delta = p_effect_given_cause - p_effect

        if delta <= 0:
            print(
                f"\n[INFO] No positive causal effect detected for "
                f"'{cause} → {effect}'."
            )
            return

        # --- Normal explanation ---
        print("\n--- Causal Edge Analysis ---\n")
        
        print(f"Cause        : {cause}")
        print(f"Effect       : {effect}\n")

        print("How the number is computed:\n")

        print(
            f"1. Overall failure rate:\n"
            f"   P({effect}) = {effect_count} / {total} = {p_effect:.2f}\n"
        )

        print(
            f"2. Failure rate when the cause appears:\n"
            f"   P({effect} | {cause}) = {joint_count} / {cause_count} "
            f"= {p_effect_given_cause:.2f}\n"
        )

        print(
            f"3. Causal strength:\n"
            f"   {p_effect_given_cause:.2f} − {p_effect:.2f} = {delta:+.2f}\n"
        )

        print(
            "Interpretation:\n"
            f"When '{cause}' appears, '{effect}' happens "
            f"{abs(delta) * 100:.0f}% more often than usual."
        )