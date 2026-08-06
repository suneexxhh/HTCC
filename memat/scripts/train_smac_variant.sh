#!/bin/sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

ENV_NAME="StarCraft2"
MAP=""
ALGO="${ALGO:-memat_episode}"
EXP=""
GPU="${GPU:-0}"
SEED="${SEED:-1}"
CLOCK_BETA=""

LR="5e-4"
N_TRAINING_THREADS="16"
N_ROLLOUT_THREADS="32"
NUM_MINI_BATCH="1"
EPISODE_LENGTH="100"
SAVE_INTERVAL="100000"
USE_EVAL=1
DRY_RUN=0

NUM_ENV_STEPS=""
PPO_EPOCH=""
CLIP_PARAM=""
STACKED_FRAMES=""

show_help() {
    cat <<EOF
Usage:
  sh train_smac_variant.sh --map <map> [--algo <algorithm>] [--seed <seed>] [--gpu <gpu>] [options] [-- extra train args]

Examples:
  sh train_smac_variant.sh --map 3m --algo memat_episode --seed 1 --gpu 0 --dry-run
  sh train_smac_variant.sh --map 3m --algo nest_mat_lite --seed 1 --gpu 0 --dry-run
  sh train_smac_variant.sh --map 3m --algo nest_mat_clock --seed 1 --gpu 0 --dry-run
  sh train_smac_variant.sh --map 5m_vs_6m --algo nest_mat_clock --clock-beta 0.5 --seed 1 --gpu 0
  sh train_smac_variant.sh --map 6h_vs_8z --algo memat --seed 2 --gpu 1
  sh train_smac_variant.sh --map 3s_vs_5z --algo mat_dec -- --n_head 2

Algorithms are passed directly to train/train_smac.py. Common values:
  memat, nest_mat_lite, nest_mat_clock, nest_mat, memat_episode, mat_dec, mat_encoder, mat_decoder, mat_gru

For nest_mat_clock, omitting --clock-beta selects the map-specific value used
in Table 3 of the HTCC paper.
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --map|-m)
            MAP="$2"; shift 2 ;;
        --algo|-a)
            ALGO="$2"; shift 2 ;;
        --seed|-s)
            SEED="$2"; shift 2 ;;
        --gpu|-g)
            GPU="$2"; shift 2 ;;
        --clock-beta|--nest-clock-beta|--nest_clock_beta)
            CLOCK_BETA="$2"; shift 2 ;;
        --exp|-e)
            EXP="$2"; shift 2 ;;
        --num-env-steps)
            NUM_ENV_STEPS="$2"; shift 2 ;;
        --ppo-epoch)
            PPO_EPOCH="$2"; shift 2 ;;
        --clip-param)
            CLIP_PARAM="$2"; shift 2 ;;
        --stacked-frames)
            STACKED_FRAMES="$2"; shift 2 ;;
        --lr)
            LR="$2"; shift 2 ;;
        --rollout-threads)
            N_ROLLOUT_THREADS="$2"; shift 2 ;;
        --training-threads)
            N_TRAINING_THREADS="$2"; shift 2 ;;
        --episode-length)
            EPISODE_LENGTH="$2"; shift 2 ;;
        --num-mini-batch)
            NUM_MINI_BATCH="$2"; shift 2 ;;
        --save-interval)
            SAVE_INTERVAL="$2"; shift 2 ;;
        --no-eval)
            USE_EVAL=0; shift ;;
        --dry-run)
            DRY_RUN=1; shift ;;
        --help|-h)
            show_help; exit 0 ;;
        --)
            shift; break ;;
        -*)
            echo "[ERROR] unknown option: $1"
            show_help
            exit 1 ;;
        *)
            break ;;
    esac
done

if [ -z "$MAP" ]; then
    echo "[ERROR] --map is required."
    show_help
    exit 1
fi

case "$MAP" in
    3m|8m|MMM)
        DEFAULT_NUM_ENV_STEPS="5000000"
        DEFAULT_PPO_EPOCH="15"
        DEFAULT_CLIP_PARAM="0.2"
        DEFAULT_STACKED_FRAMES="1"
        ;;
    1c3s5z)
        DEFAULT_NUM_ENV_STEPS="5000000"
        DEFAULT_PPO_EPOCH="10"
        DEFAULT_CLIP_PARAM="0.2"
        DEFAULT_STACKED_FRAMES="1"
        ;;
    3s_vs_5z)
        DEFAULT_NUM_ENV_STEPS="5000000"
        DEFAULT_PPO_EPOCH="15"
        DEFAULT_CLIP_PARAM="0.05"
        DEFAULT_STACKED_FRAMES="4"
        ;;
    3s5z|4m_vs_5m|5m_vs_6m|8m_vs_9m|10m_vs_11m)
        DEFAULT_NUM_ENV_STEPS="5000000"
        DEFAULT_PPO_EPOCH="10"
        DEFAULT_CLIP_PARAM="0.05"
        DEFAULT_STACKED_FRAMES="1"
        ;;
    MMM2)
        DEFAULT_NUM_ENV_STEPS="10000000"
        DEFAULT_PPO_EPOCH="5"
        DEFAULT_CLIP_PARAM="0.05"
        DEFAULT_STACKED_FRAMES="1"
        ;;
    3s5z_vs_3s6z)
        DEFAULT_NUM_ENV_STEPS="20000000"
        DEFAULT_PPO_EPOCH="5"
        DEFAULT_CLIP_PARAM="0.05"
        DEFAULT_STACKED_FRAMES="1"
        ;;
    27m_vs_30m)
        DEFAULT_NUM_ENV_STEPS="10000000"
        DEFAULT_PPO_EPOCH="5"
        DEFAULT_CLIP_PARAM="0.2"
        DEFAULT_STACKED_FRAMES="1"
        ;;
    6h_vs_8z)
        DEFAULT_NUM_ENV_STEPS="10000000"
        DEFAULT_PPO_EPOCH="15"
        DEFAULT_CLIP_PARAM="0.05"
        DEFAULT_STACKED_FRAMES="1"
        ;;
    *)
        echo "[ERROR] unknown SMAC map preset: $MAP"
        echo "Use --num-env-steps, --ppo-epoch, and --clip-param only after adding this map preset."
        exit 1
        ;;
esac

if [ -z "$CLOCK_BETA" ] && [ "$ALGO" = "nest_mat_clock" ]; then
    case "$MAP" in
        1c3s5z)          CLOCK_BETA="0.25" ;;
        3s5z)            CLOCK_BETA="0.50" ;;
        5m_vs_6m)        CLOCK_BETA="0.75" ;;
        8m_vs_9m)        CLOCK_BETA="0.00" ;;
        10m_vs_11m)      CLOCK_BETA="0.25" ;;
        6h_vs_8z)        CLOCK_BETA="0.50" ;;
        3s5z_vs_3s6z)    CLOCK_BETA="1.00" ;;
        MMM2)            CLOCK_BETA="0.50" ;;
        27m_vs_30m)      CLOCK_BETA="0.50" ;;
    esac
fi

NUM_ENV_STEPS="${NUM_ENV_STEPS:-$DEFAULT_NUM_ENV_STEPS}"
PPO_EPOCH="${PPO_EPOCH:-$DEFAULT_PPO_EPOCH}"
CLIP_PARAM="${CLIP_PARAM:-$DEFAULT_CLIP_PARAM}"
STACKED_FRAMES="${STACKED_FRAMES:-$DEFAULT_STACKED_FRAMES}"

if [ -z "$EXP" ]; then
    if [ -n "$CLOCK_BETA" ]; then
        EXP="${ALGO}_b${CLOCK_BETA}_${MAP}"
    else
        EXP="${ALGO}_${MAP}"
    fi
fi

EVAL_FLAG=""
if [ "$USE_EVAL" -eq 1 ]; then
    EVAL_FLAG="--use_eval"
fi
CLOCK_BETA_ARG=""
if [ -n "$CLOCK_BETA" ]; then
    CLOCK_BETA_ARG="--nest_clock_beta ${CLOCK_BETA}"
fi

echo "env=${ENV_NAME} map=${MAP} algo=${ALGO} exp=${EXP} seed=${SEED} gpu=${GPU}"
echo "steps=${NUM_ENV_STEPS} ppo_epoch=${PPO_EPOCH} clip=${CLIP_PARAM} stacked_frames=${STACKED_FRAMES}"
if [ -n "$CLOCK_BETA" ]; then
    echo "clock_beta=${CLOCK_BETA}"
fi

if [ "$DRY_RUN" -eq 1 ]; then
    echo "CUDA_VISIBLE_DEVICES=${GPU} python train/train_smac.py --env_name ${ENV_NAME} --algorithm_name ${ALGO} --experiment_name ${EXP} --map_name ${MAP} --seed ${SEED} --n_training_threads ${N_TRAINING_THREADS} --n_rollout_threads ${N_ROLLOUT_THREADS} --num_mini_batch ${NUM_MINI_BATCH} --episode_length ${EPISODE_LENGTH} --stacked_frames ${STACKED_FRAMES} --num_env_steps ${NUM_ENV_STEPS} --lr ${LR} --ppo_epoch ${PPO_EPOCH} --clip_param ${CLIP_PARAM} --save_interval ${SAVE_INTERVAL} --use_value_active_masks ${EVAL_FLAG} ${CLOCK_BETA_ARG} $*"
    exit 0
fi

CUDA_VISIBLE_DEVICES="${GPU}" python train/train_smac.py \
    --env_name "${ENV_NAME}" \
    --algorithm_name "${ALGO}" \
    --experiment_name "${EXP}" \
    --map_name "${MAP}" \
    --seed "${SEED}" \
    --n_training_threads "${N_TRAINING_THREADS}" \
    --n_rollout_threads "${N_ROLLOUT_THREADS}" \
    --num_mini_batch "${NUM_MINI_BATCH}" \
    --episode_length "${EPISODE_LENGTH}" \
    --stacked_frames "${STACKED_FRAMES}" \
    --num_env_steps "${NUM_ENV_STEPS}" \
    --lr "${LR}" \
    --ppo_epoch "${PPO_EPOCH}" \
    --clip_param "${CLIP_PARAM}" \
    --save_interval "${SAVE_INTERVAL}" \
    --use_value_active_masks \
    $EVAL_FLAG \
    $CLOCK_BETA_ARG \
    "$@"
