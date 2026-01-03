"""
Failure tracking across time and model versions.
"""

import json
from datetime import datetime
from collections import defaultdict
from typing import List, Dict
from ..type_definitions import FailureInstance


class FailureTracker:
    """
    Track failures across model versions and time.
    
    Example:
        >>> tracker = FailureTracker()
        >>> tracker.log("gpt-4", "v1", failures)
        >>> tracker.export("failures.json")
        >>> summary = tracker.summarize()
    """
    
    def __init__(self):
        self.history: Dict[str, List[Dict]] = defaultdict(list)
    
    def log(
        self,
        model_name: str,
        model_version: str,
        failures: List[FailureInstance],
        metadata: Dict = None
    ):
        """Log failures for a model version"""
        timestamp = datetime.utcnow().isoformat()
        
        for failure in failures:
            entry = {
                "timestamp": timestamp,
                "model": model_name,
                "failure_type": failure.failure_type.value,
                "severity": failure.severity.value,
                "confidence": failure.confidence,
                "root_causes": failure.root_causes,
                "failure_id": failure.failure_id,
                "metadata": metadata or {},
            }
            self.history[model_version].append(entry)
    
    def export(self, filepath: str):
        """Export history to JSON"""
        with open(filepath, "w") as f:
            json.dump(dict(self.history), f, indent=2)
        print(f"📊 Exported failure history to {filepath}")
    
    def summarize(self) -> Dict[str, Dict]:
        """Get summary by model version"""
        summary = {}
        
        for version, entries in self.history.items():
            counts = defaultdict(int)
            for entry in entries:
                counts[entry["failure_type"]] += 1
            
            summary[version] = {
                "total_failures": len(entries),
                "by_type": dict(counts),
                "avg_confidence": sum(e["confidence"] for e in entries) / len(entries) if entries else 0,
            }
        
        return summary
