#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ -f .pangu_build.env ]]; then
  # shellcheck disable=SC1091
  source .pangu_build.env
fi

BUILD_DIR="${BUILD_DIR:-build}"
PROBLEM="${PROBLEM:-BrioWuShocktube}"
ENABLE_CUDA="${ENABLE_CUDA:-ON}"
MPI_NP="${MPI_NP:-1}"
INPUT_FILE="${INPUT_FILE:-}"
DATA_ROOT="${DATA_ROOT:-$ROOT_DIR/data}"

if [[ "$ENABLE_CUDA" == "ON" ]]; then
  default_exe="pangu.cuda"
else
  default_exe="pangu.host"
fi

EXE_PATH="${EXE_PATH:-$BUILD_DIR/pangu/src/${EXE_NAME:-$default_exe}}"

if [[ -z "$INPUT_FILE" ]]; then
  INPUT_FILE="pangu/problem/$PROBLEM/inputfile"
fi

# Parse minimal CLI options
while [[ $# -gt 0 ]]; do
  case "$1" in
    -i|--input)
      INPUT_FILE="$2"
      shift 2
      ;;
    -b|--build-dir)
      BUILD_DIR="$2"
      if [[ "$ENABLE_CUDA" == "ON" ]]; then
        default_exe="pangu.cuda"
      else
        default_exe="pangu.host"
      fi
      EXE_PATH="$BUILD_DIR/pangu/src/${EXE_NAME:-$default_exe}"
      shift 2
      ;;
    -p|--problem)
      PROBLEM="$2"
      if [[ -z "${INPUT_FILE:-}" || "$INPUT_FILE" == pangu/problem/*/inputfile ]]; then
        INPUT_FILE="pangu/problem/$PROBLEM/inputfile"
      fi
      shift 2
      ;;
    -n|--np)
      MPI_NP="$2"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    *)
      break
      ;;
  esac
done

if [[ ! -x "$EXE_PATH" ]]; then
  echo "ERROR: Executable not found or not executable: $EXE_PATH"
  echo "Hint: run ./make.sh first."
  exit 2
fi

if [[ ! -f "$INPUT_FILE" ]]; then
  echo "ERROR: Input file not found: $INPUT_FILE"
  exit 3
fi

if [[ "$EXE_PATH" != /* ]]; then
  EXE_PATH="$ROOT_DIR/$EXE_PATH"
fi

if [[ "$INPUT_FILE" != /* ]]; then
  INPUT_FILE="$ROOT_DIR/$INPUT_FILE"
fi

# Prefer an explicit problem name, but when inputfile is given, infer from its parent dir.
problem_name="$PROBLEM"
input_problem_name="$(basename "$(dirname "$INPUT_FILE")")"
if [[ -n "$input_problem_name" && "$input_problem_name" != "." && "$input_problem_name" != "/" ]]; then
  problem_name="$input_problem_name"
fi

RUN_DIR="$DATA_ROOT/$problem_name"
mkdir -p "$RUN_DIR"

echo "[run.sh] Executable: $EXE_PATH"
echo "[run.sh] Input: $INPUT_FILE"
echo "[run.sh] Run dir: $RUN_DIR"

cd "$RUN_DIR"

if [[ "$MPI_NP" -gt 1 ]]; then
  if command -v mpirun >/dev/null 2>&1; then
    exec mpirun -np "$MPI_NP" "$EXE_PATH" -i "$INPUT_FILE" "$@"
  fi
  echo "ERROR: MPI requested (np=$MPI_NP), but mpirun is not available"
  exit 4
fi

exec "$EXE_PATH" -i "$INPUT_FILE" "$@"
