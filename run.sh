#!/bin/bash
echo "🚀 Training model..."
python -m src.train
echo "📊 Evaluating model..."
python -m src.evaluate
