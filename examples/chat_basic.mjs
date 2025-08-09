'use strict';

import util from './util.mjs';
const { postJSON, getJSON } = util;

async function main() {
  console.log('Health:', await getJSON('/healthz'));

  console.log('Loading model via HF repo + file...');
  const load = await postJSON('/load_model', {
    hf_repo: 'microsoft/Phi-3-mini-4k-instruct-gguf',
    hf_file: 'Phi-3-mini-4k-instruct-q4.gguf',
    n_ctx: 4096,
    chat_format: 'chatml',
  });
  console.log('Loaded:', load);

  const messages = [
    { role: 'system', content: 'You are helpful.' },
    { role: 'user', content: 'Write a haiku about GPUs.' },
  ];
  const resp = await postJSON('/chat', {
    messages,
    max_tokens: 128,
    temperature: 0.7,
  });
  console.log('Chat response:', resp.content);

  console.log('Unloading model...');
  console.log(await postJSON('/unload_model', {}));
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});


