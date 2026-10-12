#!/bin/bash
# Learning-curve runs, queued behind the main experiments.
# Usage: queue_learning_curve.sh <python> <cnn|bert>
PY="$1"; KIND="$2"
cd "$(dirname "$0")/.."

running() { pgrep -f "src/run_$1.py" >/dev/null || { sleep 5; pgrep -f "src/run_$1.py" >/dev/null; }; }

if [ "$KIND" = cnn ]; then
  until [ -f results/bilstm__seed3.json ] || ! running neural; do sleep 30; done
  for f in 0.1 0.25 0.5; do "$PY" src/run_neural.py --models cnn-nonstatic --seeds 1 --frac $f; done
else
  until [ -f results/bert__seed3.json ] || ! running bert; do sleep 30; done
  for f in 0.1 0.25 0.5; do
    for m in finbert bert; do "$PY" src/run_bert.py --model $m --seeds 1 --frac $f; done
  done
fi
echo "learning curve ($KIND) done"
