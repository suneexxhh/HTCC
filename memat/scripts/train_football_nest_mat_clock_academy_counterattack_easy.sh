#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
CONDA_PROFILE="${CONDA_PROFILE:-/home/sgg/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-gf}"

GPU="${1:-0}"
SEED="${2:-1}"
CLOCK_BETA="${3:-0.25}"
if [ "$#" -ge 3 ]; then
    shift 3
else
    shift "$#"
fi

ENV_NAME="football"
SCENARIO="academy_counterattack_easy"
N_AGENT=4
ALGO="nest_mat_clock"
EXP="nest_mat_softclock_b${CLOCK_BETA}"
LOG_DIR="${SCRIPT_DIR}/logs_gf"
LOG_FILE="${NEST_MAT_GF_LOG_FILE:-${LOG_DIR}/${SCENARIO}_${ALGO}_seed${SEED}_b${CLOCK_BETA}_$(date +%Y%m%d_%H%M%S).log}"

mkdir -p "${LOG_DIR}"

if [ "${RUN_IN_BACKGROUND:-1}" = "1" ] && [ "${NEST_MAT_GF_BG_CHILD:-0}" != "1" ]; then
    if command -v setsid >/dev/null 2>&1; then
        NEST_MAT_GF_BG_CHILD=1 NEST_MAT_GF_LOG_FILE="${LOG_FILE}" \
            setsid nohup "${SCRIPT_PATH}" "${GPU}" "${SEED}" "${CLOCK_BETA}" "$@" </dev/null >/dev/null 2>&1 &
    else
        NEST_MAT_GF_BG_CHILD=1 NEST_MAT_GF_LOG_FILE="${LOG_FILE}" \
            nohup "${SCRIPT_PATH}" "${GPU}" "${SEED}" "${CLOCK_BETA}" "$@" </dev/null >/dev/null 2>&1 &
    fi
    echo "started pid=$! log_file=${LOG_FILE}"
    exit 0
fi

exec >> "${LOG_FILE}" 2>&1
echo "log_file=${LOG_FILE}"
echo "pid=$$"

if [ ! -f "${CONDA_PROFILE}" ]; then
    echo "[ERROR] Conda activation script not found: ${CONDA_PROFILE}" >&2
    exit 1
fi

source "${CONDA_PROFILE}"
conda activate "${CONDA_ENV}"

echo "env=${ENV_NAME} scenario=${SCENARIO} algo=${ALGO} exp=${EXP} seed=${SEED} gpu=${GPU} clock_beta=${CLOCK_BETA}"

cd "${SCRIPT_DIR}"
EVAL_ARGS=(--use_eval)
if [ "${USE_EVAL:-1}" = "0" ]; then
    EVAL_ARGS=()
fi

CUDA_VISIBLE_DEVICES="${GPU}" python train/train_football.py \
    --seed "${SEED}" \
    --env_name "${ENV_NAME}" \
    --algorithm_name "${ALGO}" \
    --experiment_name "${EXP}" \
    --scenario "${SCENARIO}" \
    --n_agent "${N_AGENT}" \
    --lr 5e-4 \
    --entropy_coef 0.01 \
    --max_grad_norm 0.5 \
    --eval_episodes 32 \
    --n_training_threads 16 \
    --n_rollout_threads 20 \
    --num_mini_batch 1 \
    --episode_length 200 \
    --eval_interval 25 \
    --num_env_steps 10000000 \
    --ppo_epoch 10 \
    --clip_param 0.05 \
    --use_value_active_masks \
    --use_policy_active_masks \
    --nest_clock_beta "${CLOCK_BETA}" \
    "${EVAL_ARGS[@]}" \
    "$@"
