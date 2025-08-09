'use strict';

const BASE_URL = process.env.AI_TOOLKIT_BASE_URL || 'http://localhost:8000';

async function postJSON(path, payload) {
  const res = await fetch(`${BASE_URL}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    let details = '';
    try {
      details = await res.text();
    } catch (_) { }
    throw new Error(`HTTP ${res.status}: ${details || res.statusText}`);
  }
  const text = await res.text();
  return JSON.parse(text || '{}');
}

async function getJSON(path) {
  const res = await fetch(`${BASE_URL}${path}`, { method: 'GET' });
  if (!res.ok) {
    let details = '';
    try {
      details = await res.text();
    } catch (_) { }
    throw new Error(`HTTP ${res.status}: ${details || res.statusText}`);
  }
  const text = await res.text();
  return JSON.parse(text || '{}');
}

function prettyBytes(numBytes) {
  const units = ['B', 'KiB', 'MiB', 'GiB', 'TiB'];
  let size = Number(numBytes || 0);
  let unitIdx = 0;
  while (size >= 1024 && unitIdx < units.length - 1) {
    size /= 1024;
    unitIdx += 1;
  }
  return `${size.toFixed(2)} ${units[unitIdx]}`;
}

export default { BASE_URL, postJSON, getJSON, prettyBytes };


