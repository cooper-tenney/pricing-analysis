#!/usr/bin/env bash
# Smoke test: run CLI dry-run and full run, verify artifacts exist in run folder.

set -e
cd "$(dirname "$0")/.."

# Create minimal CSV for testing (use feature names that won't be dropped as id-like)
# Include AccountID_int to verify it's flagged suspicious_id_like and excluded from engineering
mkdir -p /tmp/smoke_test
echo "product,quantity,region,AccountID_int,amount" > /tmp/smoke_test/data.csv
echo "A,1,North,100001,100" >> /tmp/smoke_test/data.csv
echo "A,2,North,100002,200" >> /tmp/smoke_test/data.csv
echo "B,1,South,100003,150" >> /tmp/smoke_test/data.csv
echo "B,2,South,100004,250" >> /tmp/smoke_test/data.csv
echo "A,3,North,100005,300" >> /tmp/smoke_test/data.csv
echo "B,1,North,100006,180" >> /tmp/smoke_test/data.csv
echo "A,2,South,100007,220" >> /tmp/smoke_test/data.csv

echo "=== Smoke test: CLI dry-run ==="
.venv/bin/python main.py /tmp/smoke_test/data.csv --target amount --dry-run

LATEST=$(cat artifacts/latest.txt 2>/dev/null || true)
if [ -n "$LATEST" ]; then
  echo "Latest run_id: $LATEST"
  RUN_DIR="artifacts/runs/$LATEST"
  if [ -d "$RUN_DIR" ]; then
    echo "Checking artifacts in $RUN_DIR..."
    [ -f "$RUN_DIR/ui_summary.csv" ] && echo "  OK ui_summary.csv" || echo "  MISSING ui_summary.csv"
    [ -f "$RUN_DIR/run_config.json" ] && echo "  OK run_config.json" || echo "  MISSING run_config.json"
    [ -f "$RUN_DIR/audit_report.pdf" ] && echo "  OK audit_report.pdf" || echo "  MISSING audit_report.pdf"
  fi
fi

echo ""
echo "=== Smoke test: CLI full run ==="
.venv/bin/python main.py /tmp/smoke_test/data.csv --target amount --no-report

LATEST2=$(cat artifacts/latest.txt 2>/dev/null || true)
if [ -n "$LATEST2" ]; then
  RUN_DIR2="artifacts/runs/$LATEST2"
  if [ -d "$RUN_DIR2" ]; then
    echo "Full run artifacts:"
    [ -f "$RUN_DIR2/metrics.json" ] && echo "  OK metrics.json" || echo "  MISSING metrics.json"
    [ -f "$RUN_DIR2/predictions.csv" ] && echo "  OK predictions.csv" || echo "  MISSING predictions.csv"
    [ -f "$RUN_DIR2/permutation_importance.csv" ] && echo "  OK permutation_importance.csv" || echo "  MISSING permutation_importance.csv"
    [ -f "$RUN_DIR2/residuals_report.pdf" ] && echo "  OK residuals_report.pdf" || echo "  MISSING residuals_report.pdf"
    # Verify numeric engineering: AccountID_int must be skipped, not engineered
    if [ -f "$RUN_DIR2/run_config.json" ]; then
      .venv/bin/python -c "
import json, sys
p = '$RUN_DIR2/run_config.json'
d = json.load(open(p))
eng = d.get('engineered_numeric_cols', [])
skp = d.get('skipped_numeric_cols', {})
if 'AccountID_int' in eng:
    print('FAIL: AccountID_int must NOT be in engineered_numeric_cols')
    sys.exit(1)
if 'AccountID_int' in skp:
    print('  OK AccountID_int in skipped_numeric_cols (not engineered)')
else:
    print('  WARN: AccountID_int not found in skipped_numeric_cols')
"
    fi
  fi
fi

echo ""
echo "=== Smoke test complete ==="
