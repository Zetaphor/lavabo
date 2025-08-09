from ai_toolkit.gguf.loader import GGUFModelLoader
from pydantic import BaseModel


class Person(BaseModel):
    name: str
    age: int


# Load by Hugging Face repo
loader = GGUFModelLoader.from_hf(
    repo_id="TheBloke/Llama-2-7B-GGUF", filename="llama-2-7b.Q4_K_M.gguf"
)

# Or load from local path
# loader = GGUFModelLoader.from_path("/path/to/model.gguf")

# Free-form generation
print(loader.generate("Write a short poem about summer."))

# Structured output
result = loader.generate(
    "Extract the person's name and age from: 'Alice is 32 years old'", schema=Person
)
print(result)  # -> Person(name='Alice', age=32)
