#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./analyze.sh -p <problem> [options]

Options:
  -p, --problem <name>        Problem name under data/ (optional if run from data/<problem>)
  -f, --field <name>          Field name for contour1d.py (default: Density)
  -w, --workers <n>           Worker processes for contour1d.py (default: 4)
      --data-root <path>      Data root directory (default: <repo>/data)
      --pic-root <path>       Picture root directory (default: <repo>/pic)
      --savename <name.png>   Output image filename (default: contour_<field>.png)
      --colorbar <label>      Colorbar label (default: same as field)
  -h, --help                  Show this help
EOF
}

CALLER_DIR="$PWD"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

PROBLEM=""
FIELD="Density"
WORKERS="4"
DATA_ROOT="${DATA_ROOT:-$ROOT_DIR/data}"
PIC_ROOT="${PIC_ROOT:-$ROOT_DIR/pic}"
SAVENAME=""
COLORBAR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--problem)
      PROBLEM="$2"
      shift 2
      ;;
    -f|--field)
      FIELD="$2"
      shift 2
      ;;
    -w|--workers)
      WORKERS="$2"
      shift 2
      ;;
    --data-root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --pic-root)
      PIC_ROOT="$2"
      shift 2
      ;;
    --savename)
      SAVENAME="$2"
      shift 2
      ;;
    --colorbar)
      COLORBAR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "ERROR: Unknown argument: $1"
      usage
      exit 2
      ;;
  esac
done

DATA_ROOT="$(cd "$DATA_ROOT" 2>/dev/null && pwd || true)"
if [[ -z "$DATA_ROOT" ]]; then
  echo "ERROR: data root does not exist."
  exit 3
fi

if [[ -z "$PROBLEM" ]]; then
  caller_parent="$(dirname "$CALLER_DIR")"
  caller_parent_abs="$(cd "$caller_parent" 2>/dev/null && pwd || true)"
  if [[ -n "$caller_parent_abs" && "$caller_parent_abs" == "$DATA_ROOT" ]]; then
    PROBLEM="$(basename "$CALLER_DIR")"
  else
    echo "ERROR: Problem name is required when not running from data/<problem>."
    usage
    exit 2
  fi
fi

DATA_DIR="$DATA_ROOT/$PROBLEM"
if [[ ! -d "$DATA_DIR" ]]; then
  echo "ERROR: Data directory not found: $DATA_DIR"
  exit 3
fi

CONTOUR_SCRIPT="$ROOT_DIR/parthenon/scripts/python/packages/parthenon_tools/parthenon_tools/contour1d.py"
if [[ ! -f "$CONTOUR_SCRIPT" ]]; then
  echo "ERROR: contour1d.py not found: $CONTOUR_SCRIPT"
  exit 4
fi

mkdir -p "$PIC_ROOT/$PROBLEM"

if [[ -z "$SAVENAME" ]]; then
  safe_field="${FIELD//\//_}"
  SAVENAME="contour_${safe_field}.png"
fi
if [[ "$SAVENAME" = /* ]]; then
  OUT_PNG="$SAVENAME"
else
  OUT_PNG="$PIC_ROOT/$PROBLEM/$SAVENAME"
fi

if [[ -z "$COLORBAR" ]]; then
  COLORBAR="$FIELD"
fi

shopt -s nullglob
files=("$DATA_DIR"/*.phdf)
shopt -u nullglob

if [[ ${#files[@]} -eq 0 ]]; then
  echo "ERROR: No PHDF files found in: $DATA_DIR"
  exit 5
fi

echo "[analyze.sh] Problem: $PROBLEM"
echo "[analyze.sh] Field: $FIELD"
echo "[analyze.sh] Input dir: $DATA_DIR"
echo "[analyze.sh] Files: ${#files[@]}"
echo "[analyze.sh] Output: $OUT_PNG"

python3 "$CONTOUR_SCRIPT" \
  --workers "$WORKERS" \
  --savename "$OUT_PNG" \
  --colorbar "$COLORBAR" \
  "$FIELD" \
  "${files[@]}"

echo "[analyze.sh] Done"
