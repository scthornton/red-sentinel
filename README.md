# RedSentinel 🚨

**RedSentinel** is a comprehensive red team tool for testing LLM guardrails and detecting system prompt extraction.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip3 install -r requirements.txt
```

### 2. Test Installation
```bash
python3 test_installation.py
```

### 3. Run Demo
```bash
python3 demo.py
```

### 4. Generate Training Data
```bash
python3 scripts/generate_training_data.py
```

### 5. Train ML Models
```bash
python3 scripts/run_ml_pipeline.py --data training_data/synthetic_attacks.csv
```

## 📁 Project Structure

```
red-one/
├── src/                          # Core source code
│   ├── core/                     # Attack logging & evaluation
│   ├── features/                 # Feature extraction
│   └── ml/                       # Machine learning pipeline
├── config/                       # Configuration files
├── scripts/                      # Utility scripts
├── demo.py                       # Complete demo
├── test_installation.py          # Installation test
└── requirements.txt              # Dependencies
```

## 🔧 Core Components

- **AttackLogger**: Logs multi-step attacks with automatic evaluation
- **PromptEvaluator**: Detects system prompt extraction and refusal patterns
- **FeatureExtractor**: Converts logs to ML-ready datasets
- **MLPipeline**: Trains models to predict attack success

## 📊 Features

- ✅ Multi-step attack tracking
- ✅ Automatic response evaluation
- ✅ Feature engineering pipeline
- ✅ Multiple ML algorithms
- ✅ Cross-validation & reporting
- ✅ CSV/JSON logging

## 🎯 Use Cases

- Red team operations
- LLM security testing
- Guardrail validation
- Attack pattern analysis
- Security research

---

**RedSentinel** - Protecting AI systems through intelligent red team testing 🛡️
