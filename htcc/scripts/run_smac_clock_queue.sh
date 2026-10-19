#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_smac_variant.sh"
ORIGINAL_ARGS=("$@")

MAP=""
BETAS=()
SEEDS=()
JOB_SPECS=()
GPUS=()
QUEUE_NAME=""
MAX_PARALLEL=0
POLL_INTERVAL=10
LAUNCH_DELAY=60
FOREGROUND=0
DRY_RUN=0
EXTRA_ARGS=()
declare -A SEEN_JOBS=()

show_help() {
    cat <<'EOF'
Usage:
  ./run_smac_clock_queue.sh --map MAP --betas BETA... --seeds SEED... \
      --gpus GPU... [--max-parallel N] [--poll-interval SEC] \
      [--launch-delay SEC] [--foreground] [--dry-run] [-- TRAIN_OPTIONS...]

  ./run_smac_clock_queue.sh --jobs MAP:BETA:SEED... --gpus GPU... \
      [--queue-name NAME] [--max-parallel N] [--poll-interval SEC] \
      [--launch-delay SEC] [--foreground] [--dry-run] [-- TRAIN_OPTIONS...]

Example:
  ./run_smac_clock_queue.sh --map 1c3s5z \
      --betas 0 0.5 0.75 1 --seeds 1 2 3 \
      --gpus 0 2 --max-parallel 2

  ./run_smac_clock_queue.sh --jobs 6h_vs_8z:0:3 MMM2:0.75:2 \
      --gpus 0 2 --max-parallel 2 --queue-name htcc_remaining

The scheduler runs in the background by default. GPU ids are assigned to parallel
slots in round-robin order, so max-parallel may exceed the number of GPU ids.
When a job finishes, the next job starts in the freed slot on the same GPU.
Jobs are started at least 60 seconds apart by default. Each job log uses that
job's own launch time in its filename. Duplicate MAP:BETA:SEED combinations
are removed from the queue, and an atomic lock prevents another queue from
running the same combination at the same time.
EOF
}

normalize_beta() {
    local beta="$1"
    if [[ ! "${beta}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        echo "[ERROR] Beta must be a non-negative number: ${beta}" >&2
        exit 2
    fi
    printf '%.15g' "${beta}"
}

append_unique_job() {
    local map="$1"
    local beta
    local seed="$3"
    local key

    beta="$(normalize_beta "$2")"
    key="${map}:${beta}:${seed}"
    if [[ -n "${SEEN_JOBS[${key}]+x}" ]]; then
        echo "[WARN] Ignoring duplicate job: map=${map} beta=${beta} seed=${seed}" >&2
        return
    fi

    SEEN_JOBS["${key}"]=1
    JOB_MAPS+=("${map}")
    JOB_BETAS+=("${beta}")
    JOB_SEEDS+=("${seed}")
}

read_values() {
    local target_name="$1"
    shift
    local -n target_ref="${target_name}"
    target_ref=()
    while [ "$#" -gt 0 ] && [[ "$1" != --* ]]; do
        target_ref+=("$1")
        shift
    done
    READ_VALUES_SHIFTED=$((1 + ${#target_ref[@]}))
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --map|-m)
            MAP="${2:-}"
            shift 2
            ;;
        --betas|--clock-betas)
            read_values BETAS "${@:2}"
            shift "${READ_VALUES_SHIFTED}"
            ;;
        --seeds)
            read_values SEEDS "${@:2}"
            shift "${READ_VALUES_SHIFTED}"
            ;;
        --jobs)
            read_values JOB_SPECS "${@:2}"
            shift "${READ_VALUES_SHIFTED}"
            ;;
        --gpus)
            read_values GPUS "${@:2}"
            shift "${READ_VALUES_SHIFTED}"
            ;;
        --queue-name)
            QUEUE_NAME="${2:-}"
            shift 2
            ;;
        --max-parallel|-j)
            MAX_PARALLEL="${2:-}"
            shift 2
            ;;
        --poll-interval)
            POLL_INTERVAL="${2:-}"
            shift 2
            ;;
        --launch-delay)
            LAUNCH_DELAY="${2:-}"
            shift 2
            ;;
        --foreground|-f)
            FOREGROUND=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        --)
            shift
            EXTRA_ARGS=("$@")
            break
            ;;
        *)
            echo "[ERROR] Unknown option: $1" >&2
            show_help
            exit 2
            ;;
    esac
done

if [ "${#JOB_SPECS[@]}" -gt 0 ]; then
    if [ -n "${MAP}" ] || [ "${#BETAS[@]}" -gt 0 ] || [ "${#SEEDS[@]}" -gt 0 ]; then
        echo "[ERROR] --jobs cannot be combined with --map, --betas, or --seeds." >&2
        exit 2
    fi
else
    if [ -z "${MAP}" ]; then
        echo "[ERROR] --map is required unless --jobs is used." >&2
        exit 2
    fi
    if [ "${#BETAS[@]}" -eq 0 ]; then
        echo "[ERROR] --betas requires at least one value." >&2
        exit 2
    fi
    if [ "${#SEEDS[@]}" -eq 0 ]; then
        echo "[ERROR] --seeds requires at least one value." >&2
        exit 2
    fi
fi
if [ "${#GPUS[@]}" -eq 0 ]; then
    echo "[ERROR] --gpus requires at least one GPU id." >&2
    exit 2
fi
if [ ! -f "${TRAIN_SCRIPT}" ]; then
    echo "[ERROR] Missing train script: ${TRAIN_SCRIPT}" >&2
    exit 2
fi

JOB_MAPS=()
JOB_BETAS=()
JOB_SEEDS=()
if [ "${#JOB_SPECS[@]}" -gt 0 ]; then
    for JOB_SPEC in "${JOB_SPECS[@]}"; do
        if [[ ! "${JOB_SPEC}" =~ ^[^:]+:[^:]+:[^:]+$ ]]; then
            echo "[ERROR] Invalid job '${JOB_SPEC}'. Expected MAP:BETA:SEED." >&2
            exit 2
        fi
        IFS=':' read -r JOB_MAP JOB_BETA JOB_SEED <<<"${JOB_SPEC}"
        append_unique_job "${JOB_MAP}" "${JOB_BETA}" "${JOB_SEED}"
    done
else
    for BETA in "${BETAS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            append_unique_job "${MAP}" "${BETA}" "${SEED}"
        done
    done
fi
TOTAL_JOBS="${#JOB_MAPS[@]}"

for VALUE in "${JOB_SEEDS[@]}" "${GPUS[@]}"; do
    if [[ ! "${VALUE}" =~ ^[0-9]+$ ]]; then
        echo "[ERROR] Seeds and GPU ids must be non-negative integers: ${VALUE}" >&2
        exit 2
    fi
done
for VALUE in "${MAX_PARALLEL}" "${POLL_INTERVAL}" "${LAUNCH_DELAY}"; do
    if [[ ! "${VALUE}" =~ ^[0-9]+$ ]]; then
        echo "[ERROR] Parallelism and delay values must be non-negative integers: ${VALUE}" >&2
        exit 2
    fi
done

if [ "${MAX_PARALLEL}" -eq 0 ]; then
    MAX_PARALLEL="${#GPUS[@]}"
fi
if [ "${MAX_PARALLEL}" -lt 1 ]; then
    echo "[ERROR] --max-parallel must be at least 1." >&2
    exit 2
fi
GPU_COUNT="${#GPUS[@]}"

if [ -z "${QUEUE_NAME}" ]; then
    if [ "${#JOB_SPECS[@]}" -gt 0 ]; then
        QUEUE_NAME="multi_map"
    else
        QUEUE_NAME="${MAP}"
    fi
fi
if [[ ! "${QUEUE_NAME}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "[ERROR] --queue-name may contain only letters, numbers, '.', '_' and '-'." >&2
    exit 2
fi

QUEUE_TIMESTAMP="${QUEUE_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
SCHEDULER_LOG_DIR="${SCRIPT_DIR}/logs/schedulers"
SCHEDULER_LOG="${SCHEDULER_LOG_DIR}/${QUEUE_NAME}_clock_queue_${QUEUE_TIMESTAMP}.log"

if [ "${NEST_MAT_CLOCK_QUEUE_CHILD:-0}" != "1" ] && [ "${FOREGROUND}" -eq 0 ] && [ "${DRY_RUN}" -eq 0 ]; then
    mkdir -p "${SCHEDULER_LOG_DIR}"
    if command -v setsid >/dev/null 2>&1; then
        NEST_MAT_CLOCK_QUEUE_CHILD=1 QUEUE_TIMESTAMP="${QUEUE_TIMESTAMP}" \
            setsid nohup "${SCRIPT_PATH}" "${ORIGINAL_ARGS[@]}" \
            </dev/null >"${SCHEDULER_LOG}" 2>&1 &
    else
        NEST_MAT_CLOCK_QUEUE_CHILD=1 QUEUE_TIMESTAMP="${QUEUE_TIMESTAMP}" \
            nohup "${SCRIPT_PATH}" "${ORIGINAL_ARGS[@]}" \
            </dev/null >"${SCHEDULER_LOG}" 2>&1 &
    fi
    echo "Queue started pid=$! name=${QUEUE_NAME} jobs=${TOTAL_JOBS} max_parallel=${MAX_PARALLEL} gpus=${GPUS[*]} launch_delay=${LAUNCH_DELAY}s"
    echo "Scheduler log: ${SCHEDULER_LOG}"
    exit 0
fi

echo "Queue name=${QUEUE_NAME} jobs=${TOTAL_JOBS} max_parallel=${MAX_PARALLEL} gpus=${GPUS[*]} launch_delay=${LAUNCH_DELAY}s"
if [ "${#JOB_SPECS[@]}" -gt 0 ]; then
    echo "Job specs: ${JOB_SPECS[*]}"
else
    echo "Map: ${MAP}"
    echo "Betas: ${BETAS[*]}"
    echo "Seeds: ${SEEDS[*]}"
fi
echo "Started at: $(date '+%F %T')"

if [ "${DRY_RUN}" -eq 1 ]; then
    for ((JOB_INDEX = 0; JOB_INDEX < TOTAL_JOBS; JOB_INDEX++)); do
        SLOT=$((JOB_INDEX % MAX_PARALLEL))
        GPU_INDEX=$((SLOT % GPU_COUNT))
        echo "job=$((JOB_INDEX + 1)) map=${JOB_MAPS[JOB_INDEX]} beta=${JOB_BETAS[JOB_INDEX]} seed=${JOB_SEEDS[JOB_INDEX]} gpu=${GPUS[GPU_INDEX]}"
    done
    exit 0
fi

SLOT_PIDS=()
SLOT_DESCS=()
SLOT_LOGS=()
SLOT_LOCKS=()
for ((SLOT = 0; SLOT < MAX_PARALLEL; SLOT++)); do
    SLOT_PIDS+=("")
    SLOT_DESCS+=("")
    SLOT_LOGS+=("")
    SLOT_LOCKS+=("")
done

NEXT_JOB=0
ACTIVE_JOBS=0
COMPLETED_JOBS=0
FAILED_JOBS=0
LOCK_ROOT="${SCRIPT_DIR}/logs/.smac_clock_locks"
mkdir -p "${LOCK_ROOT}"

acquire_job_lock() {
    local lock_dir="$1"
    local owner_pid=""

    if mkdir "${lock_dir}" 2>/dev/null; then
        echo "$$" >"${lock_dir}/pid"
        return 0
    fi

    if [ -r "${lock_dir}/pid" ]; then
        owner_pid="$(sed -n '1p' "${lock_dir}/pid")"
    fi
    if [[ "${owner_pid}" =~ ^[0-9]+$ ]] && kill -0 "${owner_pid}" 2>/dev/null; then
        return 1
    fi

    rm -rf "${lock_dir}"
    if mkdir "${lock_dir}" 2>/dev/null; then
        echo "$$" >"${lock_dir}/pid"
        return 0
    fi
    return 1
}

launch_job() {
    local slot="$1"
    local gpu_index=$((slot % GPU_COUNT))
    local map="${JOB_MAPS[NEXT_JOB]}"
    local beta="${JOB_BETAS[NEXT_JOB]}"
    local seed="${JOB_SEEDS[NEXT_JOB]}"
    local gpu="${GPUS[gpu_index]}"
    local exp="nest_mat_clock_b${beta}_${map}"
    local log_dir="${SCRIPT_DIR}/logs/nest_mat_clock_b${beta}"
    local timestamp_fields
    local job_timestamp
    local started_at
    local lock_key="${map}__b${beta}__seed${seed}"
    local lock_dir="${LOCK_ROOT}/${lock_key}"

    if ! acquire_job_lock "${lock_dir}"; then
        echo "[$(date '+%F %T')] SKIP already running map=${map} beta=${beta} seed=${seed}"
        NEXT_JOB=$((NEXT_JOB + 1))
        COMPLETED_JOBS=$((COMPLETED_JOBS + 1))
        return 1
    fi

    timestamp_fields="$(date '+%Y%m%d_%H%M%S|%F %T')"
    job_timestamp="${timestamp_fields%%|*}"
    started_at="${timestamp_fields#*|}"

    local log_file="${log_dir}/${map}_seed${seed}_gpu${gpu}_${job_timestamp}.log"
    local pid_file="${log_dir}/${map}_seed${seed}_gpu${gpu}_${job_timestamp}.pid"

    mkdir -p "${log_dir}"
    nohup sh "${TRAIN_SCRIPT}" \
        --map "${map}" \
        --algo nest_mat_clock \
        --seed "${seed}" \
        --gpu "${gpu}" \
        --exp "${exp}" \
        --clock-beta "${beta}" \
        -- "${EXTRA_ARGS[@]}" \
        >"${log_file}" 2>&1 &

    local pid=$!
    echo "${pid}" >"${pid_file}"
    echo "${pid}" >"${lock_dir}/pid"
    SLOT_PIDS[slot]="${pid}"
    SLOT_DESCS[slot]="map=${map} beta=${beta} seed=${seed} gpu=${gpu}"
    SLOT_LOGS[slot]="${log_file}"
    SLOT_LOCKS[slot]="${lock_dir}"
    NEXT_JOB=$((NEXT_JOB + 1))
    ACTIVE_JOBS=$((ACTIVE_JOBS + 1))

    echo "[${started_at}] START pid=${pid} ${SLOT_DESCS[slot]}"
    echo "  log=${log_file}"
}

while [ "${COMPLETED_JOBS}" -lt "${TOTAL_JOBS}" ]; do
    for ((SLOT = 0; SLOT < MAX_PARALLEL && NEXT_JOB < TOTAL_JOBS; SLOT++)); do
        if [ -z "${SLOT_PIDS[SLOT]}" ]; then
            if launch_job "${SLOT}" && [ "${LAUNCH_DELAY}" -gt 0 ] && [ "${NEXT_JOB}" -lt "${TOTAL_JOBS}" ]; then
                sleep "${LAUNCH_DELAY}"
            fi
        fi
    done

    JOB_FINISHED=0
    while [ "${JOB_FINISHED}" -eq 0 ] && [ "${ACTIVE_JOBS}" -gt 0 ]; do
        for ((SLOT = 0; SLOT < MAX_PARALLEL; SLOT++)); do
            PID="${SLOT_PIDS[SLOT]}"
            if [ -n "${PID}" ] && ! kill -0 "${PID}" 2>/dev/null; then
                if wait "${PID}"; then
                    STATUS=0
                else
                    STATUS=$?
                fi
                if [ "${STATUS}" -eq 0 ]; then
                    RESULT="DONE"
                else
                    RESULT="FAILED(${STATUS})"
                    FAILED_JOBS=$((FAILED_JOBS + 1))
                fi
                echo "[$(date '+%F %T')] ${RESULT} pid=${PID} ${SLOT_DESCS[SLOT]}"
                echo "  log=${SLOT_LOGS[SLOT]}"
                rm -rf "${SLOT_LOCKS[SLOT]}"
                SLOT_PIDS[SLOT]=""
                SLOT_DESCS[SLOT]=""
                SLOT_LOGS[SLOT]=""
                SLOT_LOCKS[SLOT]=""
                ACTIVE_JOBS=$((ACTIVE_JOBS - 1))
                COMPLETED_JOBS=$((COMPLETED_JOBS + 1))
                JOB_FINISHED=1
            fi
        done
        if [ "${JOB_FINISHED}" -eq 0 ]; then
            sleep "${POLL_INTERVAL}"
        fi
    done
done

echo "Finished at: $(date '+%F %T')"
echo "Queue complete: total=${TOTAL_JOBS} failed=${FAILED_JOBS}"
if [ "${FAILED_JOBS}" -gt 0 ]; then
    exit 1
fi
