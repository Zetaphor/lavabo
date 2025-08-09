'use strict';

import util from './util.mjs';
const { postJSON, getJSON } = util;

async function main() {
  console.log('Health:', await getJSON('/healthz'));

  // 1) Load an embedding model
  console.log('Loading embedding model...');
  const load = await postJSON('/embeddings/load', {
    model_name: 'sentence-transformers/all-MiniLM-L6-v2',
  });
  console.log('Loaded embedding model:', load);

  // 2) Generate embeddings for a few texts
  const texts = [
    'open the browser and search for cats',
    'play some music',
    'turn on the lights in the living room',
  ];
  console.log('Generating embeddings for:', texts);
  const gen = await postJSON('/embeddings/generate', {
    texts,
    normalize: true,
  });
  console.log('Embedding dimension:', gen.dim);
  console.log('First vector (truncated):', gen.embeddings[0].slice(0, 8), '...');

  // 3) Compute similarity between first and second vectors
  console.log('Computing similarity between vectors[0] and vectors[1]');
  const sim = await postJSON('/embeddings/similarity', {
    a: gen.embeddings[0],
    b: gen.embeddings[1],
    metric: 'cosine',
    normalize: false, // already normalized during generation
  });
  console.log('Cosine similarity:', sim.similarity.toFixed(4));

  // 4) Unload the embedding model
  console.log('Unloading embedding model...');
  console.log(await postJSON('/embeddings/unload', {}));
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});




