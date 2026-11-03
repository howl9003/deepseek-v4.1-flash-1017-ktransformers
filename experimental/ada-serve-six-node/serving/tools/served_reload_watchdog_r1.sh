#!/bin/bash
# Safety net for long bench chains driven from the desktop (Claude, 3 Nov 2026): if the driver dies, the served model must not stay down.
# Every 5 minutes: is anything on the bench? A controller or stage unit running, a verify-* or blank-cpu* unit, the bench lock held, or a GPU process.
# After 4 idle checks in a row (20 minutes) with nothing loaded, it asks for the served model through the gateway once, then exits. It also exits once a model is loaded.
# It takes no lock, touches no guard and never unloads anything.
idle=0
while true; do
  sleep 300
  if curl -s -m 5 http://127.0.0.1:8082/running | grep -q '"state"'; then echo "$(date -u +%H:%M) a model is loaded, exiting"; exit 0; fi
  busy=0
  systemctl --user list-units --type=service --state=running --no-legend "phase-g-g1v-*" "ada-stage-*" "verify-*" "blank-cpu*" "ada-serve-*" | grep -q . && busy=1
  flock -n /scratch/aster/bench/.bench.lock true || busy=1
  [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && busy=1
  if [ $busy = 1 ]; then idle=0; else idle=$((idle+1)); fi
  echo "$(date -u +%H:%M) busy=$busy idle=$idle"
  if [ $idle -ge 4 ]; then
    echo "$(date -u +%H:%M) idle 20 minutes with nothing loaded: reloading the served model"
    curl -s -m 1700 http://100.108.111.59:8080/v1/chat/completions -H "Content-Type: application/json" \
      -d '{"model":"deepseek-v4.1-flash-sglang","max_tokens":16,"temperature":0,"messages":[{"role":"user","content":"Reply with one short sentence confirming you are running."}]}' | cut -c1-200
    echo
    exit 0
  fi
done
