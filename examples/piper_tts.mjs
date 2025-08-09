'use strict';

import util from './util.mjs';
const { postJSON, getJSON } = util;

async function main() {
  const requestedVoice = 'en_US-hfc_female-medium';

  console.log('Checking Piper voices...');
  let voices = await getJSON('/piper/voices');
  console.log('Voices status:', voices);

  if (requestedVoice) {
    console.log(`Requested voice: '${requestedVoice}'`);
    const available = new Set(voices.voices || []);
    if (!available.has(requestedVoice)) {
      console.log(`Voice '${requestedVoice}' not found locally. Downloading from HF...`);
      const dl = await postJSON('/piper/download', { voice: requestedVoice });
      console.log('Downloaded:', dl);
      voices = await getJSON('/piper/voices');
      console.log('Updated voices:', voices);
    }
  } else {
    // No specific voice requested: ensure at least one voice exists
    if (!voices.voices || voices.voices.length === 0) {
      const defaultVoice = 'en_US-lessac-high';
      console.log(`No local voices found. Downloading '${defaultVoice}' from HF...`);
      const dl = await postJSON('/piper/download', { voice: defaultVoice });
      console.log('Downloaded:', dl);
      voices = await getJSON('/piper/voices');
      console.log('Updated voices:', voices);
    }
  }

  const text = 'Hello from the AI toolkit! This is a Piper text-to-speech test.';
  console.log(`\nSynthesizing text: "${text}"`);

  const voiceToUse = requestedVoice || voices.default_voice || (voices.voices && voices.voices[0]);
  const resp = await postJSON('/piper/synthesize', { text, voice: voiceToUse });

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


