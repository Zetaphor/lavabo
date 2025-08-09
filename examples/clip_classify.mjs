'use strict';

import util from './util.mjs';
const { postJSON, getJSON } = util;

// Helper: read a local image and return base64
import fs from 'node:fs/promises';
import path from 'node:path';

async function imageToBase64(absPath) {
  const data = await fs.readFile(absPath);
  return data.toString('base64');
}

async function main() {
  console.log('Health:', await getJSON('/healthz'));

  // 1) Load CLIP model
  console.log('Loading CLIP model...');
  const load = await postJSON('/clip/load', {
    model_name: 'openai/clip-vit-large-patch14',
    device: 'auto',
  });
  console.log('Loaded CLIP:', load);

  // 2) Prepare image and labels
  const imagePath = process.env.CLIP_IMAGE || path.resolve('balloon_animal.jpg');
  const imageB64 = await imageToBase64(imagePath);
  const labels = [
    'a photo of a balloon',
    'a photo of an animal',
    'a photo of a person'
  ];

  // 3) Classify
  const classify = await postJSON('/clip/classify', {
    image_base64: imageB64,
    labels,
  });
  console.log('Top-5 results:', classify.results.slice(0, 5));

  // 4) NSFW check
  const nsfw = await postJSON('/clip/nsfw', {
    image_base64: imageB64,
    threshold: 0.5,
  });
  console.log('NSFW check:', nsfw);

  // 5) Unload
  console.log('Unloading CLIP...');
  console.log(await postJSON('/clip/unload', {}));
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});


