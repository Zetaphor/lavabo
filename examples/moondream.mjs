'use strict';

import fs from 'node:fs';
import util from './util.mjs';

const IMAGE_PATH = 'balloon_animal.jpg';

async function fileToBase64(path) {
  const data = await fs.promises.readFile(path);
  return data.toString('base64');
}

async function main() {
  console.log('Health:', await util.getJSON('/healthz'));

  console.log('Loading Moondream...');
  const load = await util.postJSON('/moondream/load', {
    model_name: 'moondream/moondream-2b-2025-04-14-4bit',
    device: 'cuda',
    compile: false,
    compile_backend: 'none',
  });
  console.log('Loaded:', load);

  const imgB64 = await fileToBase64(IMAGE_PATH);

  console.log('\nShort caption:');
  const capShort = await util.postJSON('/moondream/caption', {
    image_base64: imgB64,
    length: 'short',
  });
  console.log(capShort.caption);

  console.log('\nNormal caption:');
  const capNormal = await util.postJSON('/moondream/caption', {
    image_base64: imgB64,
    length: 'normal',
  });
  console.log(capNormal.caption);

  console.log("\nVisual query: 'How many people are in the image?'");
  const vqa = await util.postJSON('/moondream/query', {
    image_base64: imgB64,
    question: 'How many people are in the image?',
  });
  console.log(vqa.answer);

  console.log("\nObject detection: 'face'");
  const det = await util.postJSON('/moondream/detect', {
    image_base64: imgB64,
    query: 'face',
  });
  console.log(`Found ${det.objects.length} face(s)`);

  console.log("\nPointing: 'person'");
  const pts = await util.postJSON('/moondream/point', {
    image_base64: imgB64,
    query: 'person',
  });
  console.log(`Found ${pts.points.length} person(s)`);

  console.log('\nUnloading Moondream...');
  console.log(await util.postJSON('/moondream/unload', {}));
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});


