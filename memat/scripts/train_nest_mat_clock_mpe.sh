#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
CONDA_PROFILE="${CONDA_PROFILE:-/home/sgg/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-new_titans}"

GPU="${1:-0}"
if [ "$#" -gt 0 ]; then
    shift
fi
CLOCK_BETA="${1:-0.25}"
if [ "$#" -gt 0 ]; then
    shift
fi

SEEDS=()
EXTRA_ARGS=()
while [ "$#" -gt 0 ]; do
    if [ "$1" = "--" ]; then
        shift
        EXTRA_ARGS=("$@")
        break
    fi
    if [[ ! "$1" =~ ^[0-9]+$ ]]; then
        echo "[ERROR] Invalid seed '$1'. Put training options after --." >&2
        echo "Usage: $0 [GPU] [BETA] [SEED ...] [-- TRAIN_OPTIONS...]" >&2
        exit 2
    fi
    SEEDS+=("$1")
    shift
done

ENV_NAME="MPE"
SCENARIO="simple_spread"
NUM_LANDMARKS=3
NUM_AGENTS=3
ALGO="nest_mat_clock"
EXP="nest_mat_softclock_b${CLOCK_BETA}"
EXP_WAS_OVERRIDDEN=0

for ((ARG_INDEX = 0; ARG_INDEX < ${#EXTRA_ARGS[@]}; ARG_INDEX++)); do
    case "${EXTRA_ARGS[ARG_INDEX]}" in
        --scenario_name)
            if ((ARG_INDEX + 1 < ${#EXTRA_ARGS[@]})); then
                SCENARIO="${EXTRA_ARGS[ARG_INDEX + 1]}"
            fi
            ;;
        --scenario_name=*) SCENARIO="${EXTRA_ARGS[ARG_INDEX]#*=}" ;;
        --num_agents)
            if ((ARG_INDEX + 1 < ${#EXTRA_ARGS[@]})); then
                NUM_AGENTS="${EXTRA_ARGS[ARG_INDEX + 1]}"
            fi
            ;;
        --num_agents=*) NUM_AGENTS="${EXTRA_ARGS[ARG_INDEX]#*=}" ;;
        --num_landmarks)
            if ((ARG_INDEX + 1 < ${#EXTRA_ARGS[@]})); then
                NUM_LANDMARKS="${EXTRA_ARGS[ARG_INDEX + 1]}"
            fi
            ;;
        --num_landmarks=*) NUM_LANDMARKS="${EXTRA_ARGS[ARG_INDEX]#*=}" ;;
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
        --nest_clock_beta)
            if ((ARG_INDEX + 1 < ${#EXTRA_ARGS[@]})); then
                CLOCK_BETA="${EXTRA_ARGS[ARG_INDEX + 1]}"
            fi
            ;;
        --nest_clock_beta=*) CLOCK_BETA="${EXTRA_ARGS[ARG_INDEX]#*=}" ;;
    esac
done

if [ "${EXP_WAS_OVERRIDDEN}" = "0" ]; then
    EXP="nest_mat_softclock_b${CLOCK_BETA}"
fi

LOG_DIR="${SCRIPT_DIR}/logs_mpe"
mkdir -p "${LOG_DIR}"

if [ "${NEST_MAT_MPE_BG_CHILD:-0}" != "1" ]; then
    if [ "${#SEEDS[@]}" -eq 0 ]; then
        SEEDS=(1 2 3)
    fi

    for SEED in "${SEEDS[@]}"; do
        LOG_SCENARIO="${SCENARIO//\//_}"
        LOG_FILE="${LOG_DIR}/${LOG_SCENARIO}_${NUM_AGENTS}a_${NUM_LANDMARKS}l_${ALGO}_seed${SEED}_b${CLOCK_BETA}_$(date +%Y%m%d_%H%M%S).log"

        if [ "${RUN_IN_BACKGROUND:-1}" = "1" ]; then
            if command -v setsid >/dev/null 2>&1; then
                NEST_MAT_MPE_BG_CHILD=1 NEST_MAT_MPE_SEED="${SEED}" NEST_MAT_MPE_BETA="${CLOCK_BETA}" NEST_MAT_MPE_LOG_FILE="${LOG_FILE}" \
                    setsid nohup "${SCRIPT_PATH}" "${GPU}" "${CLOCK_BETA}" -- "${EXTRA_ARGS[@]}" </dev/null >/dev/null 2>&1 &
            else
                NEST_MAT_MPE_BG_CHILD=1 NEST_MAT_MPE_SEED="${SEED}" NEST_MAT_MPE_BETA="${CLOCK_BETA}" NEST_MAT_MPE_LOG_FILE="${LOG_FILE}" \
                    nohup "${SCRIPT_PATH}" "${GPU}" "${CLOCK_BETA}" -- "${EXTRA_ARGS[@]}" </dev/null >/dev/null 2>&1 &
            fi
            echo "started seed=${SEED} beta=${CLOCK_BETA} pid=$! gpu=${GPU} log_file=${LOG_FILE}"
            sleep 1
        else
            NEST_MAT_MPE_BG_CHILD=1 NEST_MAT_MPE_SEED="${SEED}" NEST_MAT_MPE_BETA="${CLOCK_BETA}" NEST_MAT_MPE_LOG_FILE="${LOG_FILE}" \
                "${SCRIPT_PATH}" "${GPU}" "${CLOCK_BETA}" -- "${EXTRA_ARGS[@]}"
        fi
    done
    exit 0
fi

SEED="${NEST_MAT_MPE_SEED:?NEST_MAT_MPE_SEED is required for a child process}"
CLOCK_BETA="${NEST_MAT_MPE_BETA:?NEST_MAT_MPE_BETA is required for a child process}"
LOG_FILE="${NEST_MAT_MPE_LOG_FILE:?NEST_MAT_MPE_LOG_FILE is required for a child process}"

exec >> "${LOG_FILE}" 2>&1
echo "log_file=${LOG_FILE}"
echo "pid=$$"

if [ ! -f "${CONDA_PROFILE}" ]; then
    echo "[ERROR] Conda activation script not found: ${CONDA_PROFILE}" >&2
    exit 1
fi

source "${CONDA_PROFILE}"
conda activate "${CONDA_ENV}"

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${GPU}"

echo "conda_env=${CONDA_ENV} env=${ENV_NAME} scenario=${SCENARIO} num_agents=${NUM_AGENTS} num_landmarks=${NUM_LANDMARKS} algo=${ALGO} exp=${EXP} seed=${SEED} gpu=${GPU} clock_beta=${CLOCK_BETA}"

EVAL_ARGS=(--use_eval)
if [ "${USE_EVAL:-1}" = "0" ]; then
    EVAL_ARGS=()
fi

cd "${SCRIPT_DIR}"
exec python train/train_mpe.py \
    --env_name "${ENV_NAME}" \
    --algorithm_name "${ALGO}" \
    --experiment_name "${EXP}" \
    --scenario_name "${SCENARIO}" \
    --num_agents "${NUM_AGENTS}" \
    --num_landmarks "${NUM_LANDMARKS}" \
    --seed "${SEED}" \
    --n_block 1 \
    --n_embd 64 \
    --n_training_threads 16 \
    --n_rollout_threads 128 \
    --num_mini_batch 1 \
    --episode_length 25 \
    --num_env_steps 20000000 \
    --ppo_epoch 10 \
    --clip_param 0.05 \
    --use_ReLU \
    --gain 0.01 \
    --lr 7e-4 \
    --critic_lr 7e-4 \
    --nest_clock_beta "${CLOCK_BETA}" \
    "${EVAL_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"
