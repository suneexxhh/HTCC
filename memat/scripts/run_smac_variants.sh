#!/bin/sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_smac_variant.sh"

MAPS=""
ALGO="${ALGO:-memat_episode}"
SEED="${SEED:-1}"
SEEDS="${SEEDS:-$SEED}"
GPU="${GPU:-0}"
GPUS=""
EXP=""
CLOCK_BETA=""
FOREGROUND=0
DRY_RUN=0
LAUNCH_DELAY="${LAUNCH_DELAY:-60}"

show_help() {
    cat <<EOF
Usage:
  sh run_smac_variants.sh --map <map...> [--algo <algorithm>] [--seed <seed>|--seeds <seed...>] [--gpu <gpu>] [--gpus <gpu...>] [--launch-delay <sec>] [--foreground] [--dry-run] [-- extra train args]

Examples:
  sh run_smac_variants.sh --map 3m --algo memat_episode --seed 1 --gpu 0
  sh run_smac_variants.sh --map 3m --algo memat_episode --seeds 1 2 3 --gpu 0
  sh run_smac_variants.sh --map 5m_vs_6m --algo nest_mat_clock --clock-beta 0.5 --seed 1 --gpu 0
  sh run_smac_variants.sh --maps 4m_vs_5m 5m_vs_6m 8m_vs_9m --algo memat_episode --seeds 1 2 3 --gpu 0
  sh run_smac_variants.sh --maps 4m_vs_5m 5m_vs_6m 8m_vs_9m --gpus 0 1 2 --algo memat_episode --seeds 1 2 3
  sh run_smac_variants.sh --map 3m --algo mat_dec --foreground -- --n_head 2

By default this script starts training with nohup and writes logs/PID files under:
  memat/scripts/logs/<exp>/ if --exp is set, otherwise memat/scripts/logs/<algorithm>[_b<beta>][_p<periods>]/
Background multi-map/multi-seed launches are staggered by --launch-delay seconds, default 60.
Use --gpus to assign one GPU per map in the same order as --maps. With multiple seeds,
the same map-to-GPU assignment is reused for every seed. GPU ids may repeat.
EOF
}

count_words() {
    set -- $1
    echo "$#"
}

get_nth_word() {
    NTH="$1"
    shift
    IDX=1
    for WORD in "$@"; do
        if [ "$IDX" -eq "$NTH" ]; then
            echo "$WORD"
            return 0
        fi
        IDX=$((IDX + 1))
    done
    return 1
}

make_suffix() {
    SUFFIX=""
    if [ -n "$CLOCK_BETA" ]; then
        SUFFIX="${SUFFIX}_b${CLOCK_BETA}"
    fi
    echo "$SUFFIX"
}

make_map_exp() {
    MAP_NAME="$1"
    if [ -n "$EXP" ]; then
        echo "$EXP"
        return
    fi
    echo "${ALGO}$(make_suffix)_${MAP_NAME}"
}

make_log_group() {
    if [ -n "$EXP" ]; then
        echo "$EXP"
        return
    fi
    echo "${ALGO}$(make_suffix)"
}

while [ $# -gt 0 ]; do
    case "$1" in
        --map|--maps|-m)
            shift
            if [ $# -eq 0 ]; then
                echo "[ERROR] --map requires at least one map name."
                exit 1
            fi
            while [ $# -gt 0 ]; do
                case "$1" in
                    -*)
                        break ;;
                    *)
                        MAPS="${MAPS} $1"; shift ;;
                esac
            done
            ;;
        --algo|-a)
            ALGO="$2"; shift 2 ;;
        --seed|-s)
            SEEDS="$2"; shift 2 ;;
        --seeds|--seed-list)
            shift
            if [ $# -eq 0 ]; then
                echo "[ERROR] --seeds requires at least one seed."
                exit 1
            fi
            SEEDS=""
            while [ $# -gt 0 ]; do
                case "$1" in
                    -*)
                        break ;;
                    *)
                        SEEDS="${SEEDS} $1"; shift ;;
                esac
            done
            ;;
        --gpu|-g)
            GPU="$2"; shift 2 ;;
        --gpus|--gpu-list)
            shift
            if [ $# -eq 0 ]; then
                echo "[ERROR] --gpus requires at least one GPU id."
                exit 1
            fi
            GPUS=""
            while [ $# -gt 0 ]; do
                case "$1" in
                    -*)
                        break ;;
                    *)
                        GPUS="${GPUS} $1"; shift ;;
                esac
            done
            ;;
        --clock-beta|--nest-clock-beta|--nest_clock_beta)
            CLOCK_BETA="$2"; shift 2 ;;
        --exp|-e)
            EXP="$2"; shift 2 ;;
        --launch-delay)
            LAUNCH_DELAY="$2"; shift 2 ;;
        --foreground|-f)
            FOREGROUND=1; shift ;;
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

if [ -z "$MAPS" ]; then
    echo "[ERROR] --map is required."
    show_help
    exit 1
fi

if [ -z "$SEEDS" ]; then
    echo "[ERROR] --seeds requires at least one seed."
    exit 1
fi

case "$LAUNCH_DELAY" in
    ''|*[!0-9]*)
        echo "[ERROR] --launch-delay must be a non-negative integer."
        exit 1 ;;
esac

MAP_COUNT=$(count_words "$MAPS")
SEED_COUNT=$(count_words "$SEEDS")
TOTAL_JOB_COUNT=$((MAP_COUNT * SEED_COUNT))
GPU_COUNT=$(count_words "$GPUS")
if [ -n "$GPUS" ] && [ "$GPU_COUNT" -ne "$MAP_COUNT" ]; then
    echo "[ERROR] --gpus count (${GPU_COUNT}) must match --maps count (${MAP_COUNT})."
    echo "        maps:${MAPS}"
    echo "        gpus:${GPUS}"
    exit 1
fi

for SEED_VALUE in $SEEDS; do
    case "$SEED_VALUE" in
        ''|*[!0-9]*)
            echo "[ERROR] seed must be a non-negative integer: $SEED_VALUE"
            exit 1 ;;
    esac
done

CLOCK_BETA_ARG=""
if [ -n "$CLOCK_BETA" ]; then
    CLOCK_BETA_ARG="--clock-beta ${CLOCK_BETA}"
fi
CLOCK_PERIODS_ARG=""

if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "[ERROR] missing train script: $TRAIN_SCRIPT"
    exit 1
fi

if [ "$DRY_RUN" -eq 1 ]; then
    for SEED_VALUE in $SEEDS; do
        JOB_INDEX=1
        for MAP in $MAPS; do
            MAP_GPU="$GPU"
            if [ -n "$GPUS" ]; then
                MAP_GPU=$(get_nth_word "$JOB_INDEX" $GPUS)
            fi
            MAP_EXP="$(make_map_exp "$MAP")"
            sh "$TRAIN_SCRIPT" --map "$MAP" --algo "$ALGO" --seed "$SEED_VALUE" --gpu "$MAP_GPU" --exp "$MAP_EXP" $CLOCK_BETA_ARG --dry-run -- "$@"
            JOB_INDEX=$((JOB_INDEX + 1))
        done
    done
    exit 0
fi

if [ "$FOREGROUND" -eq 1 ]; then
    for SEED_VALUE in $SEEDS; do
        JOB_INDEX=1
        for MAP in $MAPS; do
            MAP_GPU="$GPU"
            if [ -n "$GPUS" ]; then
                MAP_GPU=$(get_nth_word "$JOB_INDEX" $GPUS)
            fi
            MAP_EXP="$(make_map_exp "$MAP")"
            sh "$TRAIN_SCRIPT" --map "$MAP" --algo "$ALGO" --seed "$SEED_VALUE" --gpu "$MAP_GPU" --exp "$MAP_EXP" $CLOCK_BETA_ARG -- "$@"
            STATUS=$?
            if [ "$STATUS" -ne 0 ]; then
                exit "$STATUS"
            fi
            JOB_INDEX=$((JOB_INDEX + 1))
        done
    done
    exit 0
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_GROUP="$(make_log_group)"
LOG_DIR="${SCRIPT_DIR}/logs/${LOG_GROUP}"
mkdir -p "$LOG_DIR"

LAUNCHED_COUNT=0
for SEED_VALUE in $SEEDS; do
    JOB_INDEX=1
    for MAP in $MAPS; do
        MAP_GPU="$GPU"
        if [ -n "$GPUS" ]; then
            MAP_GPU=$(get_nth_word "$JOB_INDEX" $GPUS)
        fi
        MAP_EXP="$(make_map_exp "$MAP")"

        LOG_FILE="${LOG_DIR}/${MAP}_seed${SEED_VALUE}_gpu${MAP_GPU}_${TIMESTAMP}.log"
        PID_FILE="${LOG_DIR}/${MAP}_seed${SEED_VALUE}_gpu${MAP_GPU}_${TIMESTAMP}.pid"

        echo "[${MAP}] algo=${ALGO} gpu=${MAP_GPU} seed=${SEED_VALUE}"
        echo "Log: ${LOG_FILE}"
        echo "PID: ${PID_FILE}"

        nohup sh "$TRAIN_SCRIPT" --map "$MAP" --algo "$ALGO" --seed "$SEED_VALUE" --gpu "$MAP_GPU" --exp "$MAP_EXP" $CLOCK_BETA_ARG -- "$@" > "$LOG_FILE" 2>&1 &

        TRAIN_PID=$!
        echo "$TRAIN_PID" > "$PID_FILE"
        echo "Started PID=${TRAIN_PID} | tail -f ${LOG_FILE}"

        LAUNCHED_COUNT=$((LAUNCHED_COUNT + 1))
        if [ "$LAUNCHED_COUNT" -lt "$TOTAL_JOB_COUNT" ] && [ "$LAUNCH_DELAY" -gt 0 ]; then
            echo "Waiting ${LAUNCH_DELAY}s before launching next job..."
            sleep "$LAUNCH_DELAY"
        fi
        JOB_INDEX=$((JOB_INDEX + 1))
    done
done
