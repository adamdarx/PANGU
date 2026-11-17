#!/usr/bin/env bash
set -euo pipefail
local_rank=${OMPI_COMM_WORLD_LOCAL_RANK:?OpenMPI local rank is required}
IFS=, read -r -a gpu_ids <<< "${CUDA_VISIBLE_DEVICES:?Slurm GPU visibility is required}"
if (( local_rank >= ${#gpu_ids[@]} )); then
  echo "GPU_BIND_ERROR local_rank=$local_rank visible=$CUDA_VISIBLE_DEVICES" >&2
  exit 80
fi
export CUDA_VISIBLE_DEVICES=${gpu_ids[$local_rank]}
unset KOKKOS_DEVICE_ID KOKKOS_NUM_DEVICES
printf 'GPU_BIND host=%s rank=%s local_rank=%s visible=%s\n' "$(hostname -s)" \
  "${OMPI_COMM_WORLD_RANK:?}" "$local_rank" "$CUDA_VISIBLE_DEVICES" >&2
start_ticks=$(awk '{print $22}' /proc/$$/stat)
printf 'STORAGE_GUARD_RANK host=%s rank=%s pid=%s start=%s\n' "$(hostname -s)" \
  "${OMPI_COMM_WORLD_RANK:?}" "$$" "$start_ticks" >&2
if [[ ${NATIVE_IO_ERRORS:-} != "" ]]; then
  export LD_PRELOAD="$NATIVE_IO_ERRORS${LD_PRELOAD:+:$LD_PRELOAD}"
fi
exec "$@"
