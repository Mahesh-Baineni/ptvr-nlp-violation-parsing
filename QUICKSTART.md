

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Place Your Data
Make sure `ptvr_reports.json` is in the parent directory or update `DATA_PATH` in `main.py`:
```python
DATA_PATH = os.path.join(BASE_DIR, "ptvr_reports.json")
```

### Step 3: Start the Server
```bash
# From the improved_pipeline directory
uvicorn main:app --reload --port 8000
```

### Step 4: Test the API

#### Health Check
```bash
curl http://localhost:8000/health
# Expected: {"status":"ok","model":"improved","version":"2.0.0"}
```

#### Get Metrics
```bash
curl http://localhost:8000/metrics
# Expected: {"accuracy":1.0,"micro_f1":1.0,"macro_f1":1.0,...}
```

#### Make Predictions
```bash
curl -X POST http://localhost:8000/predict_file \
  -F "file=@test_report.json" \
  | jq '.'
```

### Step 5: Use the Web Interface

1. Open your browser to: `file:///path/to/improved_pipeline/index.html`
2. Or start a simple HTTP server:
   ```bash
   python3 -m http.server 8080
   # Then go to: http://localhost:8080/index.html
   ```
3. Upload `ptvr_reports.json`
4. View results with confidence scores

---


### API Response Structure
```json
{
  "results": [
    {
      "guid": "E8E20579-E93A-46B1-A157-644FB4DCFD6D",
      "submitter_address": {...},
      "cases": [
        {
          "case_id": "E8E20579-...-retailer-1",
          "guid": "E8E20579-E93A-46B1-A157-644FB4DCFD6D",
          "violator_role": "retailer",
          "violator_name": "Love's Travel Stops",
          "violator_address": {...},
          "confidence": 1.0,
          "violation_description": "Sold tobacco to minors..."
        },
        {
          "case_id": "E8E20579-...-distributor-1",
          "violator_role": "distributor",
          "violator_name": "Eby-Brown Company",
          "confidence": 0.844,
          "violation_description": "Supplied cigarettes..."
        }
      ],
      "flag_for_review": false,
      "max_confidence": 1.0
    }
  ]
}
```

### Key Fields
- `confidence`: Model's confidence (0-1) for this role prediction
- `flag_for_review`: True if confidence < threshold (needs human review)
- `violation_description`: AI-extracted description of what this violator did

---

