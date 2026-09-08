#!/usr/bin/env bash
# Run LOCALLY (on the Mac) after the capture finishes and you've tarred it on the pod:
#   pod:  cd /workspace && tar czf gepard-goldens.tgz goldens_out capture.log && sha256sum gepard-goldens.tgz
#   mac:  ./fetch_results.sh <POD_IP> <PORT> [ssh-key]
#
# Downloads the tarball and unpacks it into ~/Development/_gepard-oracle/goldens/.
set -euo pipefail

IP="${1:?usage: fetch_results.sh <POD_IP> <PORT> [ssh-key]}"
PORT="${2:?usage: fetch_results.sh <POD_IP> <PORT> [ssh-key]}"
KEY="${3:-$HOME/.ssh/id_ed25519}"

DEST="$HOME/Development/_gepard-oracle"
mkdir -p "$DEST"

echo "=== downloading gepard-goldens.tgz from $IP:$PORT ==="
scp -P "$PORT" -i "$KEY" "root@$IP:/workspace/gepard-goldens.tgz" "$DEST/"

echo "=== sha256 (compare against the pod's) ==="
shasum -a 256 "$DEST/gepard-goldens.tgz"

echo "=== unpacking ==="
tar xzf "$DEST/gepard-goldens.tgz" -C "$DEST"
[ -d "$DEST/goldens" ] || mv "$DEST/goldens_out" "$DEST/goldens"

echo
echo "unpacked to $DEST/goldens"
echo "sanity: $(python3 -c "import json;m=json.load(open('$DEST/goldens/manifest.json'));print(len(m['artifacts']),'artifacts,',m['env']['torch'],'/',m['env']['transformers'])" 2>/dev/null || echo 'manifest.json missing?')"
echo "listen: open $DEST/goldens/showcase/"
echo
echo "REMINDER: go terminate the pod."
