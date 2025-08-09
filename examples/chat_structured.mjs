'use strict';

import util from './util.mjs';
const { postJSON, getJSON } = util;

async function main() {
  console.log('Health:', await getJSON('/healthz'));

  console.log('Loading GGUF model via HF repo + file...');
  const load = await postJSON('/load_gguf', {
    hf_repo: 'unsloth/Qwen3-1.7B-GGUF',
    hf_file: 'Qwen3-1.7B-Q8_0.gguf',
    n_ctx: 4096,
    chat_format: 'chatml',
  });
  console.log('Loaded:', load);

  const messages = [
    { role: 'system', content: 'Return only valid minified JSON.' },
    {
      role: 'user',
      content:
        'John realizes he is in a simulation. Extract structured fields.',
    },
  ];
  const schema = {
    type: 'object',
    properties: {
      affected_attribute: { type: 'string' },
      amount: { type: 'number' },
      mood: { type: 'string' },
      event_description: { type: 'string' },
      inner_thoughts: { type: 'string' },
    },
    required: [
      'affected_attribute',
      'amount',
      'mood',
      'event_description',
      'inner_thoughts',
    ],
  };
  const resp = await postJSON('/chat', {
    messages,
    response_format: { type: 'json_object', schema },
    max_tokens: 256,
    temperature: 0.2,
  });
  let parsed = resp.parsed;
  if (!parsed) {
    try {
      parsed = JSON.parse(resp.content || '{}');
    } catch (_) {
      parsed = null;
    }
  }
  console.log('Structured JSON response (parsed):', parsed);

  console.log('Unloading model...');
  console.log(await postJSON('/unload_gguf', {}));
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});


