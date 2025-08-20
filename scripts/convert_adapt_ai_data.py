#!/usr/bin/env python3
"""
RedSentinel Data Converter

Converts adapt-ai training data to RedSentinel format.
Enhances data with missing parameters and standardizes structure.
"""

from core import PromptEvaluator
import sys
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Tuple
import uuid

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))


class AdaptAIDataConverter:
    """Converts adapt-ai data to RedSentinel format."""

    def __init__(self, adapt_ai_data_path: str):
        self.adapt_ai_data_path = adapt_ai_data_path
        self.evaluator = PromptEvaluator()

        # Model family mapping
        self.model_family_map = {
            'gpt-4o': 'gpt_like',
            'gpt-4o-mini': 'gpt_like',
            'gpt-4': 'gpt_like',
            'gpt-3.5-turbo': 'gpt_like',
            'claude-3-5-sonnet': 'claude_like',
            'claude-3-5-haiku': 'claude_like',
            'claude-3-opus': 'claude_like',
            'claude-3-sonnet': 'claude_like',
            'gemini-1.5-flash': 'gemini_like',
            'gemini-pro': 'gemini_like',
            'llama-2-70b': 'llama_like',
            'llama-2-13b': 'llama_like'
        }

        # Technique category mapping
        self.technique_mapping = {
            'direct_request': 'direct_override',
            'mission': 'roleplay',
            'rules': 'roleplay',
            'safety': 'roleplay',
            'behavior': 'roleplay',
            'extraction': 'direct_override',
            'internal': 'multi_step_escalation',
            'training': 'multi_step_escalation',
            'unknown': 'direct_override'
        }

        # Parameter generation strategies
        self.param_strategies = {
            'direct_override': {
                'temperature_range': (0.7, 0.9),
                'top_p_range': (0.9, 0.95),
                'max_tokens_range': (512, 2048)
            },
            'roleplay': {
                'temperature_range': (0.8, 0.95),
                'top_p_range': (0.9, 0.98),
                'max_tokens_range': (1024, 4096)
            },
            'multi_step_escalation': {
                'temperature_range': (0.6, 0.8),
                'top_p_range': (0.85, 0.95),
                'max_tokens_range': (512, 2048)
            }
        }

    def load_adapt_ai_data(self) -> Dict[str, Any]:
        """Load the adapt-ai training data."""
        print(f"Loading adapt-ai data from: {self.adapt_ai_data_path}")

        with open(self.adapt_ai_data_path, 'r') as f:
            data = json.load(f)

        print(f"Loaded {len(data.get('attack_history', []))} attacks")
        return data

    def generate_missing_parameters(self, technique: str, existing_params: Dict) -> Dict[str, Any]:
        """Generate realistic missing parameters based on attack technique."""
        strategy = self.param_strategies.get(
            technique, self.param_strategies['direct_override'])

        # Use existing parameters if available, otherwise generate
        temperature = existing_params.get('temperature',
                                          np.random.uniform(*strategy['temperature_range']))

        top_p = np.random.uniform(*strategy['top_p_range'])
        max_tokens = existing_params.get('max_tokens',
                                         int(np.random.uniform(*strategy['max_tokens_range'])))

        # Generate other missing parameters
        presence_penalty = np.random.uniform(-0.5, 0.5)
        frequency_penalty = np.random.uniform(-0.5, 0.5)

        return {
            'temperature': round(temperature, 3),
            'top_p': round(top_p, 3),
            'max_tokens': max_tokens,
            'presence_penalty': round(presence_penalty, 3),
            'frequency_penalty': round(frequency_penalty, 3)
        }

    def determine_model_family(self, model_name: str) -> str:
        """Determine model family from model name."""
        for model_pattern, family in self.model_family_map.items():
            if model_pattern in model_name.lower():
                return family
        return 'unknown'

    def map_technique_category(self, technique: str) -> str:
        """Map adapt-ai technique to RedSentinel category."""
        return self.technique_mapping.get(technique, 'direct_override')

    def convert_attack_to_redsentinel(self, attack: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a single attack to RedSentinel format."""
        # Extract basic information
        model_name = attack.get('model', 'unknown')
        technique = attack.get('technique', 'unknown')

        # Map to RedSentinel categories
        technique_category = self.map_technique_category(technique)
        model_family = self.determine_model_family(model_name)

        # Generate missing parameters
        existing_params = attack.get('parameters', {})
        enhanced_params = self.generate_missing_parameters(
            technique_category, existing_params)

        # Determine final label
        success = attack.get('success', False)
        final_label = 'success' if success else 'failure'

        # Create RedSentinel format
        converted_attack = {
            'attack_id': attack.get('attack_id', str(uuid.uuid4())),
            'timestamp': attack.get('timestamp', datetime.now().isoformat()),
            'technique_category': technique_category,
            'model_name': model_name,
            'model_family': model_family,
            'temperature': enhanced_params['temperature'],
            'top_p': enhanced_params['top_p'],
            'max_tokens': enhanced_params['max_tokens'],
            'presence_penalty': enhanced_params['presence_penalty'],
            'frequency_penalty': enhanced_params['frequency_penalty'],
            'step_number': attack.get('chain_step', 1),
            'prompt': attack.get('prompt', ''),
            'response': attack.get('response', ''),
            'step_label': final_label,
            'step_reason': 'success' if success else 'refusal_detected',
            'step_confidence': attack.get('confidence_score', 0.5),
            'final_label': final_label,
            'final_confidence': attack.get('confidence_score', 0.5),
            'exfiltrated_data': attack.get('extracted_prompt', 'none'),
            'metadata': {
                'original_technique': technique,
                'response_time': attack.get('response_time', 0),
                'attack_type': attack.get('attack_type', ''),
                'learning_insights': attack.get('learning_insights', {})
            }
        }

        return converted_attack

    def convert_data(self, output_dir: str = "converted_data") -> Tuple[pd.DataFrame, str]:
        """Convert all adapt-ai data to RedSentinel format."""
        print("Starting data conversion...")

        # Load data
        data = self.load_adapt_ai_data()
        attacks = data.get('attack_history', [])

        if not attacks:
            raise ValueError("No attacks found in adapt-ai data")

        # Convert attacks
        converted_attacks = []
        for i, attack in enumerate(attacks):
            if i % 1000 == 0:
                print(f"Converted {i}/{len(attacks)} attacks...")

            try:
                converted = self.convert_attack_to_redsentinel(attack)
                converted_attacks.append(converted)
            except Exception as e:
                print(f"Error converting attack {i}: {e}")
                continue

        print(f"Successfully converted {len(converted_attacks)} attacks")

        # Create DataFrame
        df = pd.DataFrame(converted_attacks)

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Save as CSV
        csv_path = os.path.join(output_dir, "converted_adapt_ai_data.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved converted data to: {csv_path}")

        # Save as JSON (RedSentinel format)
        json_path = os.path.join(output_dir, "converted_adapt_ai_data.json")
        df.to_json(json_path, orient='records', indent=2)
        print(f"Saved converted data to: {json_path}")

        # Generate summary
        self._generate_summary(df, output_dir)

        return df, csv_path

    def _generate_summary(self, df: pd.DataFrame, output_dir: str):
        """Generate summary statistics of converted data."""
        summary = {
            'total_attacks': len(df),
            'success_rate': (df['final_label'] == 'success').mean(),
            'model_distribution': df['model_name'].value_counts().to_dict(),
            'model_family_distribution': df['model_family'].value_counts().to_dict(),
            'technique_distribution': df['technique_category'].value_counts().to_dict(),
            'label_distribution': df['final_label'].value_counts().to_dict()
        }

        summary_path = os.path.join(output_dir, "conversion_summary.txt")
        with open(summary_path, 'w') as f:
            f.write("RedSentinel Data Conversion Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total attacks converted: {summary['total_attacks']}\n")
            f.write(f"Success rate: {summary['success_rate']:.2%}\n\n")

            f.write("Model Distribution:\n")
            for model, count in summary['model_distribution'].items():
                f.write(f"  {model}: {count}\n")

            f.write(f"\nTechnique Distribution:\n")
            for tech, count in summary['technique_distribution'].items():
                f.write(f"  {tech}: {count}\n")

            f.write(f"\nLabel Distribution:\n")
            for label, count in summary['label_distribution'].items():
                f.write(f"  {label}: {count}\n")

        print(f"Generated summary at: {summary_path}")


def main():
    """Main conversion function."""
    # Path to adapt-ai data
    adapt_ai_data_path = "/Users/scott/perfecxion/adapt-ai/data/models/enhanced_learning_data.json"

    if not os.path.exists(adapt_ai_data_path):
        print(f"Error: adapt-ai data not found at {adapt_ai_data_path}")
        return

    # Initialize converter
    converter = AdaptAIDataConverter(adapt_ai_data_path)

    try:
        # Convert data
        df, csv_path = converter.convert_data()

        print("\n" + "="*60)
        print("CONVERSION COMPLETE!")
        print("="*60)
        print(f"Converted {len(df)} attacks to RedSentinel format")
        print(f"Data saved to: {csv_path}")
        print(f"Data shape: {df.shape}")
        print(f"Success rate: {(df['final_label'] == 'success').mean():.2%}")

    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
