from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import os
from typing import List
import uvicorn

# Import List from typing

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Specify GPU number

# Load model with tokenizer
model = SentenceTransformer('nvidia/NV-Embed-v2',trust_remote_code=True)

model.max_seq_length = 32768
model.tokenizer.padding_side = "right"

app = FastAPI()


class TextEmbeddingRequest(BaseModel):
    text_list: List[str]  # Use List[str] instead of list[str]
    instruction: str = ""
    max_length: int = 32768


def add_eos(input_examples):
    input_examples = [input_example + model.tokenizer.eos_token for input_example in input_examples]
    return input_examples


def get_text_embeddings(text_list, instruction="", max_length=32768):
    """
    Get vector representations of a list of texts.

    Parameters:
    text_list (List[str]): List of texts to encode.
    instruction (str): Instruction or context information for the text, default is an empty string.
    max_length (int): Maximum length of the text, default is 32768.

    Returns:
    embeddings (torch.Tensor): Vector representations of the text list.
    """
    # Combine the instruction with each query if provided
    if instruction:
        text_list = [instruction + " " + text for text in text_list]

    # Add EOS token to each text
    text_list_with_eos = add_eos(text_list)

    # Set batch size and other parameters as per original code
    batch_size = 2

    try:
        # Get the embeddings with specified parameters
        embeddings = model.encode(
            sentences=text_list_with_eos,
            batch_size=batch_size,
            normalize_embeddings=True,  # Normalize embeddings as in the original code
            convert_to_tensor=True  # Convert embeddings to tensor for similarity computation
        )

        return embeddings.tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embeddings")
async def create_embeddings(request: TextEmbeddingRequest):
    try:
        embeddings = get_text_embeddings(
            text_list=request.text_list,
            instruction=request.instruction,
            max_length=request.max_length
        )
        return {"embeddings": embeddings}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Uvicorn server.") # Add a description

    # Define the --host argument
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address to bind the server to."
    )

    # Define the --port argument
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port number to bind the server to."
    )

    # Parse the arguments
    args = parser.parse_args()

    # Use the parsed arguments in uvicorn.run
    print(f"Starting Uvicorn server on {args.host}:{args.port}") # Add a confirmation message
    uvicorn.run(app, host=args.host, port=args.port)