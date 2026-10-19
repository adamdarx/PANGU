#!/bin/bash
# Resource and transport settings follow stage02/scripts/stationary_low.slurm
# and stage02/scripts/run_case.sh. Only the output location is isolated.
#SBATCH --partition=GPU80G
#SBATCH --qos=low
#SBATCH --job-name=S02_STAT_LOW_FLOW
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
set -euo pipefail
module load gcc/12.2.0
module load cuda/12.6.0
repo_dir=/lustre/home/2601110206/PANGU
dependency_prefix=/lustre/home/2601110206
run_dir="${PANGU_WORKFLOW_RUN_DIR:?PANGU_WORKFLOW_RUN_DIR is required}"
[[ "$run_dir" == "$dependency_prefix"/pangu-workflow-* ]]
test -x "$run_dir/pangu"
test -f "$run_dir/input.in"
export PATH="$dependency_prefix/openmpi/bin:$dependency_prefix/hdf5/bin:$PATH"
export LD_LIBRARY_PATH="$dependency_prefix/openmpi/lib:$dependency_prefix/hdf5/lib:${LD_LIBRARY_PATH:-}"
export OMPI_MCA_io=ompio
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=self,smcuda,tcp
export OMPI_MCA_opal_cuda_support=true
unset AWA_FAMILY AWA_VARIANT AWA_AMPLITUDE AWA_FINAL_TIME AWA_SAVE_INITIAL AWA_INTERVAL
cd "$run_dir"
printf 'STATIONARY_LOW_BEGIN %s job=%s host=%s\n' "$(date -Is)" "$SLURM_JOB_ID" "$(hostname)"
sha256sum pangu input.in
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv
test "$(sha256sum pangu | cut -d ' ' -f1)" = "$(cut -d ' ' -f1 pangu.sha256)"
set +e
./pangu -i input.in parthenon/mesh/nx1=32 parthenon/mesh/nx2=32 parthenon/mesh/nx3=32
run_rc=$?
set -e
printf 'STATIONARY_LOW_END %s rc=%s\n' "$(date -Is)" "$run_rc"
exit "$run_rc"
