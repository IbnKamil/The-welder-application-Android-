#!/usr/bin/env bash
# Idempotent Cloud Agent bootstrap for the Сварщик ПРО repository.
# Prepares two independent toolchains:
#   1. Android SDK (35) so ./gradlew can build the Kotlin/Compose app.
#   2. A Python virtualenv for the ml/ (weldvision) research package.
# Safe to run repeatedly: every step checks for existing state before acting.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ANDROID_SDK_ROOT="${ANDROID_SDK_ROOT:-/opt/android-sdk}"
CMDLINE_TOOLS_VERSION="11076708"
PLATFORM="platforms;android-35"
BUILD_TOOLS="build-tools;35.0.0"

log() { printf '\n=== %s ===\n' "$1"; }

# --- 1. System packages -----------------------------------------------------
# Debian/Ubuntu ships python3 without ensurepip; the venv module needs it.
log "System packages"
if ! dpkg -s python3-venv >/dev/null 2>&1 && ! python3 -c 'import ensurepip' >/dev/null 2>&1; then
  sudo apt-get update -qq
  sudo apt-get install -y -qq python3-venv unzip curl
else
  # Ensure unzip/curl exist even when venv is already present.
  command -v unzip >/dev/null 2>&1 && command -v curl >/dev/null 2>&1 || {
    sudo apt-get update -qq
    sudo apt-get install -y -qq unzip curl
  }
fi

# --- 2. Android SDK ---------------------------------------------------------
log "Android SDK ($ANDROID_SDK_ROOT)"
if [ ! -d "$ANDROID_SDK_ROOT" ]; then
  sudo mkdir -p "$ANDROID_SDK_ROOT"
  sudo chown -R "$(id -u):$(id -g)" "$ANDROID_SDK_ROOT"
fi

SDKMANAGER="$ANDROID_SDK_ROOT/cmdline-tools/latest/bin/sdkmanager"
if [ ! -x "$SDKMANAGER" ]; then
  tmp_zip="$(mktemp -d)/cmdline-tools.zip"
  curl -fsSL -o "$tmp_zip" \
    "https://dl.google.com/android/repository/commandlinetools-linux-${CMDLINE_TOOLS_VERSION}_latest.zip"
  extract_dir="$(mktemp -d)"
  unzip -q "$tmp_zip" -d "$extract_dir"
  mkdir -p "$ANDROID_SDK_ROOT/cmdline-tools"
  rm -rf "$ANDROID_SDK_ROOT/cmdline-tools/latest"
  mv "$extract_dir/cmdline-tools" "$ANDROID_SDK_ROOT/cmdline-tools/latest"
fi

# Accept licenses and install required packages (sdkmanager is a no-op when present).
# `yes` is killed by SIGPIPE once sdkmanager stops reading; tolerate that under pipefail.
set +o pipefail
yes | "$SDKMANAGER" --sdk_root="$ANDROID_SDK_ROOT" --licenses >/dev/null 2>&1 || true
set -o pipefail
"$SDKMANAGER" --sdk_root="$ANDROID_SDK_ROOT" \
  "platform-tools" "$PLATFORM" "$BUILD_TOOLS" >/dev/null

# Point the Gradle build at the SDK without mutating shell profiles.
printf 'sdk.dir=%s\n' "$ANDROID_SDK_ROOT" > "$REPO_ROOT/local.properties"

# --- 3. Python ml/ package --------------------------------------------------
log "Python weldvision environment"
VENV="$REPO_ROOT/ml/.venv"
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip >/dev/null
pip install -e "$REPO_ROOT/ml[dev,export]"
deactivate

# --- 4. Warm the Gradle build ----------------------------------------------
# Downloads the Gradle distribution and Android/Compose dependencies so the
# first interactive build for an agent is fast; also validates the toolchain.
log "Gradle dependency warmup"
( cd "$REPO_ROOT" && ANDROID_SDK_ROOT="$ANDROID_SDK_ROOT" ANDROID_HOME="$ANDROID_SDK_ROOT" \
    ./gradlew --no-daemon assembleDebug )

log "Install complete"
