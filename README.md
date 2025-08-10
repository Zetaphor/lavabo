Lavabo
==============

Lavabo is an all-in-one Docker container that simplifies local AI development. It provides unified HTTP endpoints to run LLMs, text and image embedding models, vision models, and TTS models, all within a single environment.

By focusing on small, locally-hosted models, Lavabo makes powerful AI broadly accessible. The name, "Lavabo," is Spanish for washbasin, reflecting the "kitchen sink" approach to providing a complete inference solution for compact models.

Lavabo provides out-of-the-box support for a versatile range of compact models:

- **Any GGUF LLM:** Use any local or Hugging Face-hosted GGUF model for tasks ranging from standard chat to complex structured output.
- **Transformers for Embeddings:** Easily create text embeddings with built-in helper functions for similarity search.
- **Dual Text-to-Speech Engines:**
  - **Kokoro TTS:** A compact model perfect for quick integration with multiple built-in voices.
  - **Piper TTS:** Offers a large selection of voices and accents and supports training new voices for maximum flexibility.
- **CLIP for Image Classification:** Categorize images using simple and intuitive natural language prompts.
- **Moondream for Advanced Vision:** Go further with a powerful Vision-Language Model (VLM) for image captioning, visual Q&A, object detection, and pointing.

Quick start
-----------

- Build: `docker compose -f server/docker-compose.yml build`
- Run: `docker compose -f server/docker-compose.yml up`