#!/bin/bash
# LIBERO-Plus launcher: runs run_libero_manager.py in the LIBERO-Plus env
# (LIBERO_CONFIG_PATH=~/.libero_plus, MAGICK_HOME for wand). All arguments are forwarded:
#   bash experiments/libero/run_libero_plus.sh --config-name=eval_libero_plus_ditb
# Summarize with experiments/libero/summarize_results_plus.py --output_dir <dir>.

set -e

ENV_NAME=${ROBOACT_PLUS_ENV:-roboact_plus}
# conda may not be on a non-interactive shell's PATH (`conda info --base` fails), and the install
# root differs per machine (miniforge3 / miniconda3 / anaconda3). Probe the candidates in order and
# fail loudly instead of silently falling back to a python that does not exist. ROBOACT_PLUS_PY
# skips the probing entirely.
if [ -n "${ROBOACT_PLUS_PY:-}" ]; then
    PY="$ROBOACT_PLUS_PY"
    CONDA_BASE=$(dirname "$(dirname "$(dirname "$(dirname "$PY")")")")
else
    PY=""
    for base in "$(conda info --base 2>/dev/null)" "$CONDA_PREFIX_1" "$HOME/miniforge3" "$HOME/miniconda3" "$HOME/anaconda3"; do
        [ -z "$base" ] && continue
        if [ -x "$base/envs/$ENV_NAME/bin/python" ]; then
            CONDA_BASE="$base"
            PY="$base/envs/$ENV_NAME/bin/python"
            break
        fi
    done
    if [ -z "$PY" ]; then
        echo "Error: cannot find the python of conda env '$ENV_NAME'." >&2
        echo "       Tried: conda info --base / \$CONDA_PREFIX_1 / ~/miniforge3 / ~/miniconda3 / ~/anaconda3." >&2
        echo "       Run 'conda activate $ENV_NAME' first, or set ROBOACT_PLUS_PY=/path/to/envs/$ENV_NAME/bin/python." >&2
        exit 1
    fi
fi

export LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH:-$HOME/.libero_plus}
export MAGICK_HOME=${MAGICK_HOME:-$CONDA_BASE/envs/$ENV_NAME}
export LD_LIBRARY_PATH="$CONDA_BASE/envs/$ENV_NAME/lib:${LD_LIBRARY_PATH:-}"

echo "ENV=$ENV_NAME  LIBERO_CONFIG_PATH=$LIBERO_CONFIG_PATH"
echo "PYTHON=$PY"

exec "$PY" experiments/libero/run_libero_manager.py "$@"
