from chromadb.api.types import Images
from chromadb import Documents, Embeddings
from typing import Union, TypeVar, Protocol

from embedding.embeddingModel import getEmbeddings

# Define Embeddable as a union of Documents and Images
Embeddable = Union[Documents, Images]

# D is a type variable that can be either Documents or Images
D = TypeVar("D", bound=Embeddable, contravariant=True)

# Define the EmbeddingFunction protocol
class EmbeddingFunction(Protocol[D]):
    def __call__(self, input: D) -> Embeddings:
        ...

class treeQAEmbeddings(EmbeddingFunction[Documents]):

    def __call__(self, input: Documents) -> Embeddings:
        # Convert input to list if it's a single document (string)
        if isinstance(input, str):
            input = [input]  # Ensure it's in list format
        elif not isinstance(input, list):
            raise ValueError("Input should be a list of strings or a single string.")
        return getEmbeddings(input)


