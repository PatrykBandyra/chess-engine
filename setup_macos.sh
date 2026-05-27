#!/usr/bin/env bash
# =============================================================================
# Chess Engine - macOS environment setup
# =============================================================================
# Sets up everything needed to run the chess engine and experiments on macOS:
#   1. Python virtual environment (.venv at project root)
#   2. Python dependencies (requirements-macos.txt)
#   3. Stockfish binary (downloaded for detected architecture)
#   4. Updates engine/constants.py with the correct macOS Stockfish path
#
# Usage:
#     chmod +x setup_macos.sh
#     ./setup_macos.sh
#
# Optional flags:
#     ./setup_macos.sh --skip-deps        # skip Python package install
#     ./setup_macos.sh --skip-stockfish   # skip Stockfish download
#     ./setup_macos.sh --python python3.13  # use different interpreter (3.11–3.13 supported)
#
# Note: pinned deps (matplotlib/numpy/pandas) require Python 3.11–3.13 (no wheels for 3.14+).
# =============================================================================

set -euo pipefail

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
STOCKFISH_PARENT_DIR="$PROJECT_DIR/stockfish_ai"
VENV_DIR="$PROJECT_DIR/.venv"
REQUIREMENTS="$PROJECT_DIR/requirements-macos.txt"
CONSTANTS_FILE="$PROJECT_DIR/engine/constants.py"

# Stockfish 17 release URLs
SF_RELEASE_TAG="sf_17"
SF_BASE_URL="https://github.com/official-stockfish/Stockfish/releases/download/${SF_RELEASE_TAG}"

# Defaults
PYTHON_CMD="python3.12"
SKIP_DEPS=0
SKIP_STOCKFISH=0
SKIP_PATCH=0

# ----------------------------------------------------------------------------
# Parse arguments
# ----------------------------------------------------------------------------

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-deps)        SKIP_DEPS=1 ;;
        --skip-stockfish)   SKIP_STOCKFISH=1 ;;
        --skip-patch)       SKIP_PATCH=1 ;;
        --python)           PYTHON_CMD="$2"; shift ;;
        -h|--help)
            sed -n '2,30p' "$0"
            exit 0
            ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
    shift
done

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

info()    { echo -e "\033[1;36m[INFO]\033[0m  $*"; }
ok()      { echo -e "\033[1;32m[ OK ]\033[0m  $*"; }
warn()    { echo -e "\033[1;33m[WARN]\033[0m  $*"; }
err()     { echo -e "\033[1;31m[FAIL]\033[0m  $*" >&2; }

# ----------------------------------------------------------------------------
# Sanity checks
# ----------------------------------------------------------------------------

info "Project dir:       $PROJECT_DIR"
info "Parent dir:        $PARENT_DIR"
info "Stockfish target:  $STOCKFISH_PARENT_DIR"
info "venv dir:          $VENV_DIR"
echo

# OS check
if [[ "$(uname -s)" != "Darwin" ]]; then
    warn "Not running on macOS (uname=$(uname -s)). This script is intended for macOS only."
fi

# Architecture
ARCH="$(uname -m)"
case "$ARCH" in
    arm64)
        SF_FILE="stockfish-macos-m1-apple-silicon"
        ;;
    x86_64)
        SF_FILE="stockfish-macos-x86-64-modern"
        ;;
    *)
        err "Unsupported architecture: $ARCH (expected arm64 or x86_64)"
        exit 1
        ;;
esac
ok "Detected architecture: $ARCH -> binary: $SF_FILE"

# Python check — pinned dependencies (matplotlib 3.10.3, numpy 2.2.5, pandas 2.2.3)
# only ship prebuilt wheels for Python 3.11–3.13. Python 3.14+ forces source builds
# which fail (freetype download over SSL, etc.). Stick to 3.12 unless overridden.
if ! command -v "$PYTHON_CMD" >/dev/null 2>&1; then
    err "Python not found: $PYTHON_CMD"
    err "Install via: brew install python@3.12"
    err "Or pass a specific interpreter: ./setup_macos.sh --python python3.13"
    exit 1
fi

PY_VERSION="$("$PYTHON_CMD" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
PY_MAJOR="${PY_VERSION%%.*}"
PY_MINOR="${PY_VERSION##*.}"

if [[ "$PY_MAJOR" -lt 3 ]] || [[ "$PY_MAJOR" -eq 3 && "$PY_MINOR" -lt 11 ]]; then
    err "Python 3.11+ required (found $PY_VERSION)"
    err "Install via: brew install python@3.12"
    exit 1
fi

if [[ "$PY_MAJOR" -gt 3 ]] || [[ "$PY_MAJOR" -eq 3 && "$PY_MINOR" -gt 13 ]]; then
    err "Python $PY_VERSION is too new for the pinned requirements (matplotlib/numpy/pandas)."
    err "Pinned wheels only exist for Python 3.11–3.13."
    err "Install via: brew install python@3.12"
    err "Then re-run: ./setup_macos.sh --python python3.12"
    exit 1
fi
ok "Python: $PY_VERSION ($PYTHON_CMD)"

# curl check
if ! command -v curl >/dev/null 2>&1; then
    err "curl not found (should be pre-installed on macOS)"
    exit 1
fi

# ----------------------------------------------------------------------------
# Step 1: Python virtual environment
# ----------------------------------------------------------------------------

echo
info "=== Step 1: Python virtual environment ==="

if [[ -d "$VENV_DIR" ]]; then
    ok "venv already exists at $VENV_DIR (reusing)"
else
    info "Creating venv at $VENV_DIR..."
    "$PYTHON_CMD" -m venv "$VENV_DIR"
    ok "venv created"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# ----------------------------------------------------------------------------
# Step 2: Install Python dependencies
# ----------------------------------------------------------------------------

echo
if [[ "$SKIP_DEPS" -eq 1 ]]; then
    info "=== Step 2: Skipped (--skip-deps) ==="
else
    info "=== Step 2: Python dependencies ==="

    info "Upgrading pip, setuptools, wheel..."
    pip install --upgrade pip setuptools wheel >/dev/null

    if [[ -f "$REQUIREMENTS" ]]; then
        info "Installing from $REQUIREMENTS..."
        pip install -r "$REQUIREMENTS"
    else
        warn "$REQUIREMENTS not found, installing essentials directly"
        pip install \
            chess==1.11.2 \
            python-chess==1.999 \
            pygame==2.6.1 \
            stockfish==3.28.0 \
            numpy==2.2.5 \
            pandas==2.2.3 \
            matplotlib==3.10.3 \
            scipy==1.15.3
    fi
    ok "Python packages installed"

    # Sanity import test
    info "Verifying imports..."
    python -c "import chess; import chess.engine; import pygame; from stockfish import Stockfish; import pandas; import numpy; import scipy; import matplotlib; print('  All imports OK')"
fi

# ----------------------------------------------------------------------------
# Step 3: Stockfish binary
# ----------------------------------------------------------------------------

echo
if [[ "$SKIP_STOCKFISH" -eq 1 ]]; then
    info "=== Step 3: Skipped (--skip-stockfish) ==="
else
    info "=== Step 3: Stockfish binary ==="

    mkdir -p "$STOCKFISH_PARENT_DIR"
    BIN_PATH="$STOCKFISH_PARENT_DIR/stockfish/$SF_FILE"

    if [[ -x "$BIN_PATH" ]]; then
        ok "Stockfish binary already present at $BIN_PATH"
    else
        SF_URL="${SF_BASE_URL}/${SF_FILE}.tar"
        TAR_PATH="$STOCKFISH_PARENT_DIR/${SF_FILE}.tar"

        info "Downloading $SF_URL..."
        curl -L --fail --progress-bar -o "$TAR_PATH" "$SF_URL" || {
            err "Download failed. Try manual download from:"
            err "  https://stockfishchess.org/download/"
            exit 1
        }

        info "Extracting to $STOCKFISH_PARENT_DIR..."
        tar -xf "$TAR_PATH" -C "$STOCKFISH_PARENT_DIR"
        rm "$TAR_PATH"

        if [[ ! -x "$BIN_PATH" ]]; then
            # Some tar layouts may use different subdir naming. Search for it.
            FOUND="$(find "$STOCKFISH_PARENT_DIR" -name "stockfish-macos-*" -type f -perm +111 2>/dev/null | head -1)"
            if [[ -n "$FOUND" ]]; then
                warn "Expected binary at $BIN_PATH but found at: $FOUND"
                BIN_PATH="$FOUND"
            else
                err "Could not locate extracted Stockfish binary"
                exit 1
            fi
        fi

        chmod +x "$BIN_PATH"
        ok "Stockfish installed at: $BIN_PATH"
    fi

    # Test Stockfish responds to UCI
    info "Testing Stockfish responds to UCI..."
    SF_OUT="$(echo -e "uci\nquit" | "$BIN_PATH" 2>&1 | head -3 || true)"
    if echo "$SF_OUT" | grep -q "Stockfish"; then
        ok "Stockfish UCI test passed: $(echo "$SF_OUT" | head -1)"
    else
        warn "Stockfish UCI test inconclusive (output: $SF_OUT)"
    fi
fi

# ----------------------------------------------------------------------------
# Step 4: Patch engine/constants.py with macOS Stockfish path
# ----------------------------------------------------------------------------

echo
if [[ "$SKIP_PATCH" -eq 1 ]]; then
    info "=== Step 4: Skipped (--skip-patch) ==="
else
    info "=== Step 4: Patch STOCKFISH_PATH in constants.py ==="

    if [[ ! -f "$CONSTANTS_FILE" ]]; then
        err "constants.py not found at $CONSTANTS_FILE"
        exit 1
    fi

    # The engine/main.py runs from engine/ directory, so paths in constants.py
    # are relative to engine/. The Stockfish lives at PROJECT_DIR/stockfish_ai/
    # (sibling to engine/), which from chess-engine/engine/ is ../stockfish_ai/.
    NEW_PATH="../stockfish_ai/stockfish/$SF_FILE"

    # Backup once
    if [[ ! -f "$CONSTANTS_FILE.bak" ]]; then
        cp "$CONSTANTS_FILE" "$CONSTANTS_FILE.bak"
        ok "Created backup: $CONSTANTS_FILE.bak"
    fi

    # Replace STOCKFISH_PATH line using BSD sed compatible syntax (macOS native)
    python - <<PYEOF
import re
from pathlib import Path

p = Path("$CONSTANTS_FILE")
text = p.read_text(encoding='utf-8')
new = re.sub(
    r"^STOCKFISH_PATH\s*=\s*.*\$",
    "STOCKFISH_PATH = '$NEW_PATH'",
    text,
    flags=re.MULTILINE
)
p.write_text(new, encoding='utf-8')
print("  Patched STOCKFISH_PATH -> $NEW_PATH")
PYEOF

    ok "constants.py updated"
fi

# ----------------------------------------------------------------------------
# Done
# ----------------------------------------------------------------------------

echo
echo -e "\033[1;32m==============================================================\033[0m"
echo -e "\033[1;32m  Setup complete!\033[0m"
echo -e "\033[1;32m==============================================================\033[0m"
echo
echo "To run the engine:"
echo "  source $VENV_DIR/bin/activate"
echo "  cd $PROJECT_DIR/engine"
echo "  python main.py -w MINIMAX_TRAD -b MINIMAX_TRAD -m B -dw 3 -db 3 -g test_game.txt -l test_log.txt"
echo
echo "For experiments (require PowerShell Core / pwsh):"
echo "  brew install --cask powershell"
echo "  pwsh experiments/exp1/run_exp1_calibrate.ps1"
echo
echo "Stockfish binary: $STOCKFISH_PARENT_DIR/stockfish/$SF_FILE"
echo "Python venv:      $VENV_DIR"
echo
