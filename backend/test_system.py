#!/usr/bin/env python3
"""
Simple test script to verify the improved model works correctly.
Run this after installation to ensure everything is set up properly.
"""

import json
import sys
from pathlib import Path

print("SYSTEM VERIFICATION TEST")

# Test 1: Check imports
print("TEST 1: Checking imports...")
from backend.ml_model import ImprovedViolationMLModel
from backend.pipeline import ViolationPipeline
from backend.agents import ParserAgent, ClassifierAgent
from backend.schemas import PredictFileResponse
import sklearn
import numpy



# ----------------- TEST 2: data file -----------------
print("\nTEST 2: Checking data file...")

BASE_DIR = Path(__file__).resolve().parent
data_path = BASE_DIR / "ptvr_reports.json"

if not data_path.exists():
    print(f"Data file not found: {data_path}")
    print("   Make sure ptvr_reports.json is inside the backend/ folder.")
    sys.exit(1)

with open(data_path, "r") as f:
    data = json.load(f)
    num_reports = len(data.get("reports", []))
    print(f"Data file found: {num_reports} reports")

# Test 3: Train model
print("\nTEST 3: Training model (this may take 10-30 seconds)...")
try:
    model = ImprovedViolationMLModel(data_path=str(data_path), use_ensemble=False)
    model.train()
    print("Model trained successfully")
except Exception as e:
    print(f"Training failed: {e}")
    sys.exit(1)

# Test 4: Check metrics
print("\nTEST 4: Checking model metrics...")
metrics = model.metrics
accuracy = metrics.get('accuracy', 0)
micro_f1 = metrics.get('micro_f1', 0)
macro_f1 = metrics.get('macro_f1', 0)

print(f"  Accuracy:  {accuracy:.3f}")
print(f"  Micro F1:  {micro_f1:.3f}")
print(f"  Macro F1:  {macro_f1:.3f}")

if accuracy >= 0.95:
    print("Model performance is excellent")
elif accuracy >= 0.80:
    print("Model performance is good but could be better")
else:
    print("Model performance is below expectations")
    print("Check your data quality and feature engineering")

# Test 5: Test prediction
print("\nTEST 5: Testing prediction on sample report...")
try:
    test_report = data['reports'][0]
    roles, probs, conf = model.predict_roles_for_report(test_report)
    
    print(f"  Predicted roles: {roles}")
    print(f"  Confidence scores:")
    for role, prob in probs.items():
        print(f"    - {role}: {prob:.3f}")
    print(f"  Max confidence: {conf:.3f}")
    
    # Check ground truth
    violators = test_report['Violation'].get('Violator', [])
    true_roles = set()
    for v in violators:
        role_id = v.get('violatorTypeIDs')
        if role_id == 1:
            true_roles.add('retailer')
        elif role_id == 2:
            true_roles.add('distributor')
        elif role_id == 3:
            true_roles.add('manufacturer')
    
    print(f"  Ground truth: {sorted(true_roles)}")
    
    if set(roles) == true_roles:
        print("Prediction matches ground truth perfectly")
    else:
        print(" Prediction differs from ground truth")
        print(f"   Predicted: {set(roles)}")
        print(f"   Expected:  {true_roles}")
        
except Exception as e:
    print(f"Prediction failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Test pipeline
print("\nTEST 6: Testing full pipeline...")
try:
    pipeline = ViolationPipeline(model)
    state = pipeline.process_report(test_report)
    
    cases = state.get('cases', [])
    print(f"  Generated {len(cases)} case(s)")
    
    for i, case in enumerate(cases, 1):
        print(f"  Case {i}:")
        print(f"    - Role: {case['violator_role']}")
        print(f"    - Name: {case['violator_name']}")
        print(f"    - Confidence: {case['confidence']:.3f}")
    
    if len(cases) > 0:
        print("Pipeline executed successfully")
    else:
        print(" Pipeline generated no cases")
        
except Exception as e:
    print(f"Pipeline test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Check file structure
print("\nTEST 7: Checking file structure...")
required_files = [
    'ml_model.py',
    'pipeline.py',
    'agents.py',
    'schemas.py',
    'main.py',
    'index.html',
    'requirements.txt',
    'README.md',
]

missing_files = []
for file in required_files:
    if not Path(file).exists():
        missing_files.append(file)

if missing_files:
    print(f"Missing files: {', '.join(missing_files)}")
else:
    print("All required files present")

# Summary
print("\n" + "="*70)
print("TEST SUMMARY")

all_tests_passed = (
    accuracy >= 0.95 and
    len(cases) > 0 and
    not missing_files
)

if all_tests_passed:
    print(" ALL TESTS PASSED! System is ready to use.")
    print()
    print("Next steps:")
    print("1. Start the server: uvicorn main:app --reload")
    print("2. Open browser to: http://localhost:8000/docs")
    print("3. Try the web interface: open index.html")
    print()
else:
    print("Some tests had warnings. Review output above.")
    print()
    


sys.exit(0 if all_tests_passed else 1)
