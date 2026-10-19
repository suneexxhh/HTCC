#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
CONDA_PROFILE="${CONDA_PROFILE:-/home/sgg/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-new_titans}"

GPU="${1:-0}"
SEED="${2:-1}"
CLOCK_BETA="${3:-0.25}"
if [ "$#" -ge 3 ]; then
    shift 3
else
    shift "$#"
fi

ENV_NAME="mujoco"
SCENARIO="HalfCheetah-v2"
AGENT_CONF="6x1"
AGENT_OBSK=0
FAULTY_NODE=-1
EVAL_FAULTY_NODE=-1
ALGO="nest_mat_clock"
EXP="nest_mat_softclock_b${CLOCK_BETA}"

EXTRA_ARGS=("$@")
for ((ARG_INDEX = 0; ARG_INDEX < ${#EXTRA_ARGS[@]}; ARG_INDEX++)); do
    case "${EXTRA_ARGS[ARG_INDEX]}" in
        --scenario)
            if ((ARG_INDEX + 1 < ${#EXTRA_ARGS[@]})); then
                SCENARIO="${EXTRA_ARGS[ARG_INDEX + 1]}"
            fi
            ;;
        --scenario=*)
            SCENARIO="${EXTRA_ARGS[ARG_INDEX]#*=}"
            ;;
        --agent_conf)
            if ((ARG_INDEX + 1 < ${#EXTRA_ARGS[@]})); then
                AGENT_CONF="${EXTRA_ARGS[ARG_INDEX + 1]}"
            fi
            ;;
        --agent_conf=*)
            AGENT_CONF="${EXTRA_ARGS[ARG_INDEX]#*=}"
            ;;
    esac
done

LOG_SCENARIO="${SCENARIO//\//_}"
LOG_AGENT_CONF="${AGENT_CONF//|/_}"
LOG_DIR="${SCRIPT_DIR}/logs_mujoco"
LOG_FILE="${NEST_MAT_MUJOCO_LOG_FILE:-${LOG_DIR}/${LOG_SCENARIO}_${LOG_AGENT_CONF}_${ALGO}_seed${SEED}_b${CLOCK_BETA}_$(date +%Y%m%d_%H%M%S).log}"

mkdir -p "${LOG_DIR}"

if [ "${RUN_IN_BACKGROUND:-1}" = "1" ] && [ "${NEST_MAT_MUJOCO_BG_CHILD:-0}" != "1" ]; then
    if command -v setsid >/dev/null 2>&1; then
        NEST_MAT_MUJOCO_BG_CHILD=1 NEST_MAT_MUJOCO_LOG_FILE="${LOG_FILE}" \
            setsid nohup "${SCRIPT_PATH}" "${GPU}" "${SEED}" "${CLOCK_BETA}" "$@" </dev/null >/dev/null 2>&1 &
    else
        NEST_MAT_MUJOCO_BG_CHILD=1 NEST_MAT_MUJOCO_LOG_FILE="${LOG_FILE}" \
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

echo "conda_env=${CONDA_ENV} env=${ENV_NAME} scenario=${SCENARIO} agent_conf=${AGENT_CONF} algo=${ALGO} exp=${EXP} seed=${SEED} gpu=${GPU} clock_beta=${CLOCK_BETA}"

cd "${SCRIPT_DIR}"
EVAL_ARGS=(--use_eval)
if [ "${USE_EVAL:-1}" = "0" ]; then
    EVAL_ARGS=()
fi

CUDA_VISIBLE_DEVICES="${GPU}" python train/train_mujoco.py \
    --seed "${SEED}" \
    --env_name "${ENV_NAME}" \
    --algorithm_name "${ALGO}" \
    --experiment_name "${EXP}" \
    --scenario "${SCENARIO}" \
    --agent_conf "${AGENT_CONF}" \
    --agent_obsk "${AGENT_OBSK}" \
    --faulty_node "${FAULTY_NODE}" \
    --eval_faulty_node "${EVAL_FAULTY_NODE}" \
    --critic_lr 5e-5 \
    --lr 5e-5 \
    --entropy_coef 0.001 \
    --max_grad_norm 0.5 \
    --eval_episodes 5 \
    --n_training_threads 16 \
    --n_rollout_threads 40 \
    --num_mini_batch 40 \
    --episode_length 100 \
    --eval_interval 25 \
    --num_env_steps 10000000 \
    --ppo_epoch 10 \
    --clip_param 0.05 \
    --add_center_xy \
    --use_state_agent \
    --use_value_active_masks \
    --use_policy_active_masks \
    --nest_clock_beta "${CLOCK_BETA}" \
    "${EVAL_ARGS[@]}" \
    "$@"
