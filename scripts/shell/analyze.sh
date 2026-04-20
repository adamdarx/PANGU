#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./analyze.sh -p <problem> [options]

Options:
  -p, --problem <name>        Problem name under data/ (optional if run from data/<problem>)
  -f, --field <name>          Field name for contour1d.py (default: density)
  -w, --workers <n>           Worker processes for contour1d.py (default: 4)
      --movie2d               Use movie2d.py to generate 2D frame images
      --xzplot                Use xzplot.py to generate x-z transformed frame images
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
FIELD="density"
WORKERS="4"
USE_MOVIE2D="OFF"
USE_XZPLOT="OFF"
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
    --movie2d)
      USE_MOVIE2D="ON"
      shift
      ;;
    --xzplot)
      USE_XZPLOT="ON"
      shift
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
MOVIE2D_SCRIPT="$ROOT_DIR/parthenon/scripts/python/packages/parthenon_tools/parthenon_tools/movie2d.py"
XZPLOT_SCRIPT="$ROOT_DIR/scripts/python/xzplot.py"

if [[ "$USE_MOVIE2D" == "ON" && "$USE_XZPLOT" == "ON" ]]; then
  echo "ERROR: --movie2d and --xzplot are mutually exclusive"
  exit 2
fi

extract_metric_param() {
  local file_path="$1"
  local key="$2"
  awk -F'=' -v wanted_key="$key" '
    tolower($0) ~ /^\s*<metric>\s*$/ {in_metric=1; next}
    /^\s*</ {if (in_metric) in_metric=0}
    in_metric {
      k=$1; v=$2;
      gsub(/[[:space:]]/, "", k);
      gsub(/[[:space:]]/, "", v);
      k=tolower(k);
      if (k == wanted_key) {print v; exit}
    }
  ' "$file_path"
}

metric_file="$ROOT_DIR/pangu/problem/$PROBLEM/inputfile"
kerr_a="0.9375"
kerr_h="0.0"
if [[ -f "$metric_file" ]]; then
  parsed_a="$(extract_metric_param "$metric_file" "a" || true)"
  parsed_h="$(extract_metric_param "$metric_file" "h" || true)"
  if [[ -n "$parsed_a" ]]; then
    kerr_a="$parsed_a"
  fi
  if [[ -n "$parsed_h" ]]; then
    kerr_h="$parsed_h"
  fi
fi

mkdir -p "$PIC_ROOT/$PROBLEM"

if [[ -z "$SAVENAME" ]]; then
  safe_field="${FIELD//\//_}"
  if [[ "$USE_MOVIE2D" == "ON" ]]; then
    SAVENAME="movie2d_${safe_field}"
  elif [[ "$USE_XZPLOT" == "ON" ]]; then
    SAVENAME="xzplot_${safe_field}"
  else
    SAVENAME="contour_${safe_field}.png"
  fi
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
if [[ "$USE_MOVIE2D" == "ON" ]]; then
  echo "[analyze.sh] Mode: movie2d"
  echo "[analyze.sh] Output dir: $OUT_PNG"
elif [[ "$USE_XZPLOT" == "ON" ]]; then
  echo "[analyze.sh] Mode: xzplot"
  echo "[analyze.sh] Output dir: $OUT_PNG"
  echo "[analyze.sh] METRIC a=$kerr_a h=$kerr_h"
else
  echo "[analyze.sh] Mode: contour1d"
  echo "[analyze.sh] Output: $OUT_PNG"
fi

if [[ "$USE_MOVIE2D" == "ON" ]]; then
  if [[ ! -f "$MOVIE2D_SCRIPT" ]]; then
    echo "ERROR: movie2d.py not found: $MOVIE2D_SCRIPT"
    exit 4
  fi
  mkdir -p "$OUT_PNG"
  python3 "$MOVIE2D_SCRIPT" \
    --workers "$WORKERS" \
    --output-directory "$OUT_PNG" \
    --prefix "${FIELD//\//_}" \
    "$FIELD" \
    "${files[@]}"
elif [[ "$USE_XZPLOT" == "ON" ]]; then
  if [[ ! -f "$XZPLOT_SCRIPT" ]]; then
    echo "ERROR: xzplot.py not found: $XZPLOT_SCRIPT"
    exit 4
  fi
  mkdir -p "$OUT_PNG"
  python3 "$XZPLOT_SCRIPT" \
    --workers "$WORKERS" \
    --output-directory "$OUT_PNG" \
    --prefix "${FIELD//\//_}" \
    --kerr-a "$kerr_a" \
    --kerr-h "$kerr_h" \
    "$FIELD" \
    "${files[@]}"
else
  if [[ ! -f "$CONTOUR_SCRIPT" ]]; then
    echo "ERROR: contour1d.py not found: $CONTOUR_SCRIPT"
    exit 4
  fi
  python3 "$CONTOUR_SCRIPT" \
    --workers "$WORKERS" \
    --savename "$OUT_PNG" \
    --colorbar "$COLORBAR" \
    "$FIELD" \
    "${files[@]}"
fi

echo "[analyze.sh] Done"
