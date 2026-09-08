#!/usr/bin/env bash
# Gepard golden-capture pod bootstrap. Run ON THE RUNPOD POD (as root) after
# uploading this kit to /workspace/oracle-capture:
#
#   bash /workspace/oracle-capture/pod_bootstrap.sh
#
# Idempotent — safe to re-run if a step fails.
set -euo pipefail

KIT=/workspace/oracle-capture
REPO=/workspace/gepard-inference

echo "=== [1/4] system deps (apt) ==="
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y --no-install-recommends git git-lfs curl ffmpeg ca-certificates
git lfs install --skip-repo || true

echo "=== [2/4] uv ==="
if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
grep -q '.local/bin' "$HOME/.bashrc" || echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
uv --version

echo "=== [3/4] clone gepard-inference + build its inference venv ==="
if [ ! -d "$REPO" ]; then
  git clone https://github.com/nineninesix-ai/gepard-inference.git "$REPO"
fi
cd "$REPO"
# Their one-shot env builder: uv-provisioned Python 3.12 venv, CUDA-matched torch,
# nemo-toolkit[tts], transformers==5.3.0 re-pin, torchcodec ABI fix. (== `make setup`)
bash scripts/setup.sh

echo "=== [4/4] install the capture kit into the repo ==="
cp -f "$KIT/capture_goldens.py" "$KIT/texts.json" "$REPO/"
mkdir -p "$REPO/extra_ref_audio"
if compgen -G "$KIT/extra_ref_audio/*.wav" > /dev/null; then
  cp -f "$KIT"/extra_ref_audio/*.wav "$REPO/extra_ref_audio/"
  echo "extra ref clips: $(ls "$REPO"/extra_ref_audio/)"
else
  echo "NOTE: no extra_ref_audio/*.wav uploaded — capture will use only the repo's demo clips."
fi

echo
echo "🎉 Bootstrap done. Next:"
echo "  cd $REPO && source venv/bin/activate"
echo "  python capture_goldens.py --out /workspace/goldens_out 2>&1 | tee /workspace/capture.log"
