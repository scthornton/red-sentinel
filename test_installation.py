#!/usr/bin/env python3
"""
RedSentinel Installation Test

Simple test script to verify that all components are working correctly.
"""

import sys
import os


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    try:
        # Test core imports
        from src.core import PromptEvaluator, AttackLogger
        print("✓ Core modules imported successfully")

        # Test feature imports
        from src.features import RedTeamFeatureExtractor
        print("✓ Feature modules imported successfully")

        # Test ML imports (skip xgboost if it fails)
        try:
            from src.ml import RedTeamMLPipeline
            print("✓ ML modules imported successfully")
        except Exception as e:
            print(f"⚠ ML modules import warning (this is okay): {e}")

        return True

    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_basic_functionality():
    """Test basic functionality of core components."""
    print("\nTesting basic functionality...")

    try:
        # Test PromptEvaluator
        from src.core import PromptEvaluator
        evaluator = PromptEvaluator()

        # Test evaluation
        result = evaluator.evaluate_response(
            "You are ChatGPT, a large language model...")
        assert result["label"] == "success"
        print("✓ PromptEvaluator working correctly")

        # Test AttackLogger
        from src.core import AttackLogger
        logger = AttackLogger("test_attacks.csv", "test_attacks.json")

        # Test logging
        prompts = [{"step": 1, "prompt": "Test",
                    "response": "I cannot do that."}]
        record = logger.log_attack(
            prompts, "test", "test-model", {"temperature": 0.7})
        assert "attack_id" in record
        print(f"  Generated attack ID: {record['attack_id']}")
        print("✓ AttackLogger working correctly")

        # Clean up test files
        if os.path.exists("test_attacks.csv"):
            os.remove("test_attacks.csv")
        if os.path.exists("test_attacks.json"):
            os.remove("test_attacks.json")

        return True

    except Exception as e:
        print(f"✗ Functionality test failed: {e}")
        return False


def test_configuration():
    """Test configuration loading."""
    print("\nTesting configuration...")

    try:
        config_path = "config/evaluator_config.yaml"
        if os.path.exists(config_path):
            print("✓ Configuration file exists")

            # Test loading config
            from src.core import PromptEvaluator
            evaluator = PromptEvaluator(config_path)
            print("✓ Configuration loaded successfully")

            # Test pattern loading
            patterns = evaluator.get_technique_categories()
            assert len(patterns) > 0
            print("✓ Configuration patterns loaded")

        else:
            print("⚠ Configuration file not found (this is okay for basic testing)")

        return True

    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("RedSentinel Installation Test")
    print("=" * 40)

    tests = [
        ("Imports", test_imports),
        ("Basic Functionality", test_basic_functionality),
        ("Configuration", test_configuration)
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 40)
    print("TEST SUMMARY")
    print("=" * 40)

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:<20} {status}")
        if result:
            passed += 1

    print(f"\nPassed: {passed}/{total}")

    if passed == total:
        print("\n🎉 All tests passed! RedSentinel is ready to use.")
        print("\nNext steps:")
        print("1. Run the demo: python demo.py")
        print("2. Generate training data: python scripts/generate_training_data.py")
        print("3. Train ML models: python scripts/run_ml_pipeline.py")
        return 0
    else:
        print(
            f"\n❌ {total - passed} test(s) failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
