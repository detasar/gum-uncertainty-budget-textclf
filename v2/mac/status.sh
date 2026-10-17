#!/bin/bash
cd ~/uq-budget-v2; date; ls -la raw raw_xplat 2>/dev/null | grep npz; for f in raw/schema_cache/*/*.jsonl; do [ -f "$f" ] && echo "$f $(wc -l < "$f")"; done
echo "--- torch log"; tail -8 logs/torch.log; echo "--- clef log"; tail -4 logs/clef.log; pgrep -fl "score_|run_torch|run_clef" | cut -c1-150
