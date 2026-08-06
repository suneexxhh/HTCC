#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
CONDA_PROFILE="${CONDA_PROFILE:-/home/sgg/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-bidh}"
ISAAC_GYM_PYTHON="${ISAAC_GYM_PYTHON:-/home/sgg/pkg/isaacgym/python}"

GPU="${1:-0}"
SEED="${2:-1}"
CLOCK_BETA="${3:-0.25}"
TASK="${4:-ShadowHandCatchOver2Underarm}"
if [ "$#" -ge 4 ]; then
    shift 4
else
    shift "$#"
fi

ENV_NAME="hands"
ALGO="nest_mat_clock"
EXP="nest_mat_softclock_b${CLOCK_BETA}"
EXP_WAS_OVERRIDDEN=0
GAMMA=0.96

EXTRA_ARGS=("$@")
for ((ARG_INDEX = 0; ARG_INDEX < ${#EXTRA_ARGS[@]}; ARG_INDEX++)); do
    case "${EXTRA_ARGS[ARG_INDEX]}" in
        --task)
            if ((ARG_INDEX + 1 < ${#EXTRA_ARGS[@]})); then
                TASK="${EXTRA_ARGS[ARG_INDEX + 1]}"
            fi
            ;;
        --task=*)
            TASK="${EXTRA_ARGS[ARG_INDEX]#*=}"
            ;;
        --experiment_name)
            if ((ARG_INDEX + 1 < ${#EXTRA_ARGS[@]})); then
                EXP="${EXTRA_ARGS[ARG_INDEX + 1]}"
                EXP_WAS_OVERRIDDEN=1
            fi
            ;;
        --experiment_name=*)
            EXP="${EXTRA_ARGS[ARG_INDEX]#*=}"
            EXP_WAS_OVERRIDDEN=1
            ;;
        --gamma)
            if ((ARG_INDEX + 1 < ${#EXTRA_ARGS[@]})); then
                GAMMA="${EXTRA_ARGS[ARG_INDEX + 1]}"
            fi
            ;;
        --gamma=*)
            GAMMA="${EXTRA_ARGS[ARG_INDEX]#*=}"
            ;;
        --nest_clock_beta)
            if ((ARG_INDEX + 1 < ${#EXTRA_ARGS[@]})); then
                CLOCK_BETA="${EXTRA_ARGS[ARG_INDEX + 1]}"
            fi
            ;;
        --nest_clock_beta=*)
            CLOCK_BETA="${EXTRA_ARGS[ARG_INDEX]#*=}"
            ;;
    esac
done

if [ "${EXP_WAS_OVERRIDDEN}" = "0" ]; then
    EXP="nest_mat_softclock_b${CLOCK_BETA}"
fi

LOG_TASK="${TASK//\//_}"
LOG_DIR="${SCRIPT_DIR}/logs_hands"
LOG_FILE="${NEST_MAT_HANDS_LOG_FILE:-${LOG_DIR}/${LOG_TASK}_${ALGO}_seed${SEED}_b${CLOCK_BETA}_$(date +%Y%m%d_%H%M%S).log}"

mkdir -p "${LOG_DIR}"

if [ "${RUN_IN_BACKGROUND:-1}" = "1" ] && [ "${NEST_MAT_HANDS_BG_CHILD:-0}" != "1" ]; then
    if command -v setsid >/dev/null 2>&1; then
        NEST_MAT_HANDS_BG_CHILD=1 NEST_MAT_HANDS_LOG_FILE="${LOG_FILE}" \
            setsid nohup "${SCRIPT_PATH}" "${GPU}" "${SEED}" "${CLOCK_BETA}" "${TASK}" "$@" </dev/null >/dev/null 2>&1 &
    else
        NEST_MAT_HANDS_BG_CHILD=1 NEST_MAT_HANDS_LOG_FILE="${LOG_FILE}" \
            nohup "${SCRIPT_PATH}" "${GPU}" "${SEED}" "${CLOCK_BETA}" "${TASK}" "$@" </dev/null >/dev/null 2>&1 &
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

if [ ! -d "${ISAAC_GYM_PYTHON}/isaacgym" ]; then
    echo "[ERROR] Isaac Gym Python package not found: ${ISAAC_GYM_PYTHON}/isaacgym" >&2
    exit 1
fi

source "${CONDA_PROFILE}"
conda activate "${CONDA_ENV}"

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${ISAAC_GYM_PYTHON}${PYTHONPATH:+:${PYTHONPATH}}"

echo "conda_env=${CONDA_ENV} env=${ENV_NAME} task=${TASK} algo=${ALGO} exp=${EXP} seed=${SEED} gpu=${GPU} gamma=${GAMMA} clock_beta=${CLOCK_BETA}"

cd "${SCRIPT_DIR}"
CUDA_VISIBLE_DEVICES="${GPU}" python train/train_hands.py \
    --env_name "${ENV_NAME}" \
    --seed "${SEED}" \
    --algorithm_name "${ALGO}" \
    --experiment_name "${EXP}" \
    --task "${TASK}" \
    --n_rollout_threads 80 \
    --lr 5e-5 \
    --entropy_coef 0.001 \
    --max_grad_norm 0.5 \
    --log_interval 25 \
    --n_training_threads 16 \
    --num_mini_batch 1 \
    --num_env_steps 50000000 \
    --gamma "${GAMMA}" \
    --ppo_epoch 5 \
    --clip_param 0.2 \
    --use_value_active_masks \
    --add_center_xy \
    --use_state_agent \
    --use_policy_active_masks \
    --nest_clock_beta "${CLOCK_BETA}" \
    --headless \
    "$@"
