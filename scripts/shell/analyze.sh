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
        --cks                   Use xzplot_cks.py for Cartesian CKS x-z frame images
        --2t                    Use xz_temperature_plot.py to generate ion/electron x-z frames
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
USE_CKS="OFF"
USE_2T="OFF"
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
    --cks)
      USE_CKS="ON"
      USE_XZPLOT="ON"
      shift
      ;;
    --2t)
      USE_2T="ON"
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
XZPLOT_CKS_SCRIPT="$ROOT_DIR/scripts/python/xzplot_cks.py"
XZ_TEMP_SCRIPT="$ROOT_DIR/scripts/python/xz_temperature_plot.py"

if [[ "$USE_MOVIE2D" == "ON" && "$USE_XZPLOT" == "ON" ]]; then
  echo "ERROR: --movie2d and --xzplot are mutually exclusive"
  exit 2
fi
if [[ "$USE_MOVIE2D" == "ON" && "$USE_CKS" == "ON" ]]; then
  echo "ERROR: --movie2d and --cks are mutually exclusive"
  exit 2
fi
if [[ "$USE_2T" == "ON" && ("$USE_MOVIE2D" == "ON" || "$USE_XZPLOT" == "ON") ]]; then
  echo "ERROR: --2t is mutually exclusive with --movie2d and --xzplot"
  exit 2
fi
if [[ "$USE_2T" == "ON" && "$USE_CKS" == "ON" ]]; then
  echo "ERROR: --2t is mutually exclusive with --cks"
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

mkdir -p "$PIC_ROOT"

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
    if [[ "$USE_CKS" == "ON" ]]; then
      SAVENAME="xzplot_cks_${safe_field}"
    else
      SAVENAME="xzplot_${safe_field}"
    fi
  elif [[ "$USE_2T" == "ON" ]]; then
    SAVENAME="xztemp_${safe_field}"
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
  if [[ "$USE_CKS" == "ON" ]]; then
    echo "[analyze.sh] Mode: xzplot (CKS)"
  else
    echo "[analyze.sh] Mode: xzplot"
  fi
  echo "[analyze.sh] Output dir: $OUT_PNG"
  if [[ "$USE_CKS" == "ON" ]]; then
    echo "[analyze.sh] METRIC a=$kerr_a"
  else
    echo "[analyze.sh] METRIC a=$kerr_a h=$kerr_h"
  fi
elif [[ "$USE_2T" == "ON" ]]; then
  echo "[analyze.sh] Mode: 2T xz temperature frames"
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
  if [[ "$USE_CKS" == "ON" ]]; then
    if [[ ! -f "$XZPLOT_CKS_SCRIPT" ]]; then
      echo "ERROR: xzplot_cks.py not found: $XZPLOT_CKS_SCRIPT"
      exit 4
    fi
  else
    if [[ ! -f "$XZPLOT_SCRIPT" ]]; then
      echo "ERROR: xzplot.py not found: $XZPLOT_SCRIPT"
      exit 4
    fi
  fi
  mkdir -p "$OUT_PNG"
  if [[ "$USE_CKS" == "ON" ]]; then
    python3 "$XZPLOT_CKS_SCRIPT" \
      --workers "$WORKERS" \
      --output-directory "$OUT_PNG" \
      --prefix "${FIELD//\//_}" \
      --kerr-a "$kerr_a" \
      "$FIELD" \
      "${files[@]}"
  else
    python3 "$XZPLOT_SCRIPT" \
      --workers "$WORKERS" \
      --output-directory "$OUT_PNG" \
      --prefix "${FIELD//\//_}" \
      --kerr-a "$kerr_a" \
      --kerr-h "$kerr_h" \
      "$FIELD" \
      "${files[@]}"
  fi
elif [[ "$USE_2T" == "ON" ]]; then
  if [[ ! -f "$XZ_TEMP_SCRIPT" ]]; then
    echo "ERROR: xz_temperature_plot.py not found: $XZ_TEMP_SCRIPT"
    exit 4
  fi
  mkdir -p "$OUT_PNG"
  python3 "$XZ_TEMP_SCRIPT" \
    --workers "$WORKERS" \
    --output-directory "$OUT_PNG" \
    --prefix "${FIELD//\//_}" \
    --kerr-a "$kerr_a" \
    --kerr-h "$kerr_h" \
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
