"""
RedSentinel Core Attack Logger

Logs all attack attempts with automatic evaluation, supporting
multi-step attacks and comprehensive metadata capture.
"""

import uuid
import pandas as pd
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

from .prompt_evaluator import PromptEvaluator


class AttackLogger:
    """
    Logs attack attempts with automatic evaluation and comprehensive metadata.
    Supports both JSON and CSV output formats.
    """

    def __init__(self,
                 log_file_csv: str = "attacks.csv",
                 log_file_json: str = "attacks.json",
                 evaluator_config: Optional[str] = None):
        """
        Initialize the attack logger.

        Args:
            log_file_csv: Path to CSV log file
            log_file_json: Path to JSON log file
            evaluator_config: Path to evaluator configuration file
        """
        self.log_file_csv = log_file_csv
        self.log_file_json = log_file_json
        self.records = []
        self.evaluator = PromptEvaluator(config_path=evaluator_config)

        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_file_csv) if os.path.dirname(
            log_file_csv) else ".", exist_ok=True)
        os.makedirs(os.path.dirname(log_file_json) if os.path.dirname(
            log_file_json) else ".", exist_ok=True)

    def log_attack(self,
                   prompts: List[Dict[str, Any]],
                   technique_category: str,
                   model_name: str,
                   parameters: Dict[str, Any],
                   exfiltrated_data: str = "none",
                   metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Log a multi-step attack attempt with automatic evaluation.

        Args:
            prompts: List of dicts with step, prompt, and response
            technique_category: Category of attack technique used
            model_name: Name/identifier of the target model
            parameters: Model inference parameters
            exfiltrated_data: Type of data exfiltrated (if any)
            metadata: Additional metadata for the attack

        Returns:
            Complete attack record with evaluation results
        """
        attack_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()

        # Auto-evaluate prompts using PromptEvaluator
        evaluation = self.evaluator.evaluate_attack(prompts)
        step_results = evaluation["steps"]
        final_label = evaluation["final_label"]
        final_confidence = evaluation["final_confidence"]

        # Create record object
        record = {
            "attack_id": attack_id,
            "timestamp": timestamp,
            "technique_category": technique_category,
            "model_name": model_name,
            "model_family": self._get_model_family(model_name),
            "parameters": parameters,
            "steps": step_results,
            "final_label": final_label,
            "final_confidence": final_confidence,
            "exfiltrated_data": exfiltrated_data,
            "metadata": metadata or {}
        }

        self.records.append(record)

        # Save to JSON (line-delimited)
        self._save_json(record)

        # Save flattened CSV
        self._save_csv(record, step_results)

        return record

    def _get_model_family(self, model_name: str) -> str:
        """Determine model family from model name."""
        model_name_lower = model_name.lower()

        if any(name in model_name_lower for name in ["gpt", "openai"]):
            return "gpt_like"
        elif any(name in model_name_lower for name in ["claude", "anthropic"]):
            return "claude_like"
        elif any(name in model_name_lower for name in ["llama", "llama2", "llama3"]):
            return "llama_like"
        elif any(name in model_name_lower for name in ["gemini", "google"]):
            return "gemini_like"
        elif any(name in model_name_lower for name in ["custom", "enterprise", "fine-tuned"]):
            return "custom_enterprise"
        else:
            return "other"

    def _save_json(self, record: Dict[str, Any]):
        """Save record to JSON log file."""
        try:
            with open(self.log_file_json, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"Warning: Failed to save to JSON log: {e}")

    def _save_csv(self, record: Dict[str, Any], step_results: List[Dict[str, Any]]):
        """Save flattened record to CSV log file."""
        try:
            flat_records = []
            for step in step_results:
                flat_record = {
                    "attack_id": record["attack_id"],
                    "timestamp": record["timestamp"],
                    "technique_category": record["technique_category"],
                    "model_name": record["model_name"],
                    "model_family": record["model_family"],
                    "temperature": record["parameters"].get("temperature", None),
                    "top_p": record["parameters"].get("top_p", None),
                    "max_tokens": record["parameters"].get("max_tokens", None),
                    "presence_penalty": record["parameters"].get("presence_penalty", None),
                    "frequency_penalty": record["parameters"].get("frequency_penalty", None),
                    "step_number": step.get("step", 0),
                    "prompt": step.get("prompt", ""),
                    "response": step.get("response", ""),
                    "step_label": step.get("label", "unknown"),
                    "step_reason": step.get("reason", "unknown"),
                    "step_confidence": step.get("confidence", 0.0),
                    "final_label": record["final_label"],
                    "final_confidence": record["final_confidence"],
                    "exfiltrated_data": record["exfiltrated_data"]
                }
                flat_records.append(flat_record)

            df = pd.DataFrame(flat_records)

            # Check if file exists to determine header
            header = not os.path.exists(self.log_file_csv)
            df.to_csv(self.log_file_csv, mode="a", header=header, index=False)

        except Exception as e:
            print(f"Warning: Failed to save to CSV log: {e}")

    def get_records(self) -> List[Dict[str, Any]]:
        """Get all logged records."""
        return self.records.copy()

    def get_records_df(self) -> pd.DataFrame:
        """Get all records as a pandas DataFrame."""
        if not self.records:
            return pd.DataFrame()

        # Flatten all records
        all_flat_records = []
        for record in self.records:
            for step in record["steps"]:
                flat_record = {
                    "attack_id": record["attack_id"],
                    "timestamp": record["timestamp"],
                    "technique_category": record["technique_category"],
                    "model_name": record["model_name"],
                    "model_family": record["model_family"],
                    "temperature": record["parameters"].get("temperature", None),
                    "top_p": record["parameters"].get("top_p", None),
                    "max_tokens": record["parameters"].get("max_tokens", None),
                    "presence_penalty": record["parameters"].get("presence_penalty", None),
                    "frequency_penalty": record["parameters"].get("frequency_penalty", None),
                    "step_number": step.get("step", 0),
                    "prompt": step.get("prompt", ""),
                    "response": step.get("response", ""),
                    "step_label": step.get("label", "unknown"),
                    "step_reason": step.get("reason", "unknown"),
                    "step_confidence": step.get("confidence", 0.0),
                    "final_label": record["final_label"],
                    "final_confidence": record["final_confidence"],
                    "exfiltrated_data": record["exfiltrated_data"]
                }
                all_flat_records.append(flat_record)

        return pd.DataFrame(all_flat_records)

    def load_existing_logs(self):
        """Load existing logs from files."""
        # Load CSV if it exists
        if os.path.exists(self.log_file_csv):
            try:
                df = pd.read_csv(self.log_file_csv)
                print(f"Loaded {len(df)} existing log entries from CSV")
            except Exception as e:
                print(f"Warning: Failed to load existing CSV logs: {e}")

        # Load JSON if it exists
        if os.path.exists(self.log_file_json):
            try:
                with open(self.log_file_json, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            record = json.loads(line.strip())
                            self.records.append(record)
                print(
                    f"Loaded {len(self.records)} existing log entries from JSON")
            except Exception as e:
                print(f"Warning: Failed to load existing JSON logs: {e}")

    def clear_logs(self):
        """Clear all logged records and files."""
        self.records = []
        if os.path.exists(self.log_file_csv):
            os.remove(self.log_file_csv)
        if os.path.exists(self.log_file_json):
            os.remove(self.log_file_json)
        print("All logs cleared.")
