#!/bin/bash
# Single sequential driver (one Mac job at a time): wait for a running run_torch.sh (else run it), then Clef (200 items/dataset). Resumable.
cd ~/uq-budget-v2
if pgrep -f run_torch.sh >/dev/null; then echo "waiting for run_torch.sh $(date)"; while pgrep -f run_torch.sh >/dev/null; do sleep 30; done; else ./run_torch.sh >> logs/torch.log 2>&1; fi
echo "torch queue finished $(date); starting Clef"
echo "=== seq start $(date)" >> logs/clef.log; ./run_clef.sh >> logs/clef.log 2>&1
echo "SEQ_DONE $(date)"
