#!/bin/bash
# Record Robot State Script
#
# Usage:
#   ./run_record_robot_state.sh              # 대화형 모드
#   ./run_record_robot_state.sh 2 initial    # robot2 initial_state
#   ./run_record_robot_state.sh 3 free       # robot3 free_state

cd "$(dirname "${BASH_SOURCE[0]}")"

python scripts/record_robot_state.py "$@"
