#!/bin/bash
# LIBERO-PLUS evaluation - pass log path and task only.
# Usage: ./test_libero_plus.sh <log_path> [task]
#   log_path: dir to save log.txt and videos (required)
#   task: libero_spatial | libero_goal | libero_object | libero_10  (default: libero_spatial)

LOG_PATH=${1:?Usage: $0 <log_path> [task]}
TASK=${2:-libero_spatial}

echo "=============================="
echo "Log path: $LOG_PATH | Task: $TASK"
echo "=============================="

python libero_plus_client.py \
    --task_suites "$TASK" \
    --log_file "${LOG_PATH}/log.txt" \
    --video_log_dir "${LOG_PATH}/videos"
