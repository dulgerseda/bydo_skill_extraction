#!/bin/bash

# answers/ → normalized_answers/
echo "🔄 Step 1: Normalizing answers..."
python3 scripts/normalize_answers.py

# normalized_answers/ → results/
echo "🔄 Step 2: Running reranking pipeline..."
python3 scripts/reranking_pipeline.py

# results/ → Streamlit UI
echo "🚀 Step 3: Starting Streamlit..."
streamlit run app.py

# source .venv/bin/activate
# chmod +x run.sh && ./run.sh

