'use strict';

import util from './util.mjs';
const { postJSON, getJSON } = util;

async function main() {
  console.log('Getting available Kokoro TTS voices...');
  const voices = await getJSON('/kokoro/voices');
  console.log('Available voices:', voices);

  const text = 'Hello from the AI toolkit! This is a test of the text-to-speech system.';
  console.log(`\nSynthesizing text: "${text}"`);

  const resp = await postJSON('/kokoro/synthesize', { text });

  console.log('TTS synthesis response:', resp);
  console.log(`\nAudio available at: ${util.BASE_URL}${resp.url}`);
  console.log(
    `You can play this in your browser or use a command-line player, e.g.:\nffplay -autoexit -nodisp ${util.BASE_URL}${resp.url}`
  );
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
