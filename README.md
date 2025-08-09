AI Toolkit API
==============

Dockerized FastAPI server exposing endpoints to load/unload GGUF models and perform chat completions via `llama-cpp-python`.

Quick start
-----------

- Build: `docker compose -f server/docker-compose.yml build`
- Run: `docker compose -f server/docker-compose.yml up`
- Examples (Node 18+):
  - Basic chat: `node examples/chat_basic.mjs`
  - Structured JSON output: `node examples/chat_structured.mjs`
  - VRAM usage: `node examples/vram.mjs`

Documentation
-------------

- See `server/README.md` for detailed build/run instructions and endpoint usage.
- The example clients demonstrate free-form chat and structured JSON output using `response_format` with a JSON schema. The server returns a `parsed` field when JSON decoding succeeds.

License
-------

MIT