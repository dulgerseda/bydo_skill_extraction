#!/bin/bash

# answers_v10.json → normalized_answers_v10.json
echo "🔄 Step 1: Normalizing answers..."
python3 scripts/normalize_answers_v10.py

# normalized_answers_v10.json → results/output_v10.json
echo "🔄 Step 2: Running reranking pipeline..."
python3 scripts/reranking_pipeline.py

# results/output_v10.json → Streamlit UI
echo "🚀 Step 3: Starting Streamlit..."
streamlit run app.py


# chmod +x run.sh && ./run.sh

