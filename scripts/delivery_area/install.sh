#!/usr/bin/env bash
set -euo pipefail
src=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
dest=/autodl-fs/data/eth3d_delivery_area/tools/commands_20261128
mkdir -p "$dest" /usr/local/bin
for name in pipeline.py run.sh; do
  if [[ "$src" != "$dest" ]]; then
    if [[ -f "$dest/$name" ]] && ! cmp -s "$src/$name" "$dest/$name"; then echo "Existing different deployment: $dest/$name"; exit 1; fi
    cp "$src/$name" "$dest/$name"
  fi
done
if [[ -e /usr/local/bin/tagsda ]] && ! grep -q 'eth3d_delivery_area/tools/commands_20261128/run.sh' /usr/local/bin/tagsda; then echo 'Existing unrelated tagsda command'; exit 1; fi
printf '%s\n' '#!/usr/bin/env bash' 'exec bash /autodl-fs/data/eth3d_delivery_area/tools/commands_20261128/run.sh "$@"' > /usr/local/bin/tagsda
chmod +x /usr/local/bin/tagsda
echo 'Installed: tagsda. New terminals need no source, export, cd, or conda activation.'
