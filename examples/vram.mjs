'use strict';

import util from './util.mjs';
const { getJSON, prettyBytes } = util;

async function main() {
  console.log('Health:', await getJSON('/healthz'));

  const vram = await getJSON('/vram');
  if (!vram.gpu_available) {
    console.log('GPU/NVML is not available in the server environment.');
    return;
  }
  const devices = vram.devices || [];
  if (!devices.length) {
    console.log('No GPU devices reported.');
    return;
  }

  for (const d of devices) {
    const name = d.name || '?';
    const idx = d.index ?? -1;
    const total = d.total_bytes || 0;
    const free = d.free_bytes || 0;
    const usedByProc = d.used_by_process_bytes || 0;
    const used = total - free;
    console.log(
      `GPU ${idx} - ${name}: total=${prettyBytes(total)}, used=${prettyBytes(used)}, free=${prettyBytes(free)}, used_by_api=${prettyBytes(usedByProc)}`,
    );
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});


