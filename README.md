# TreeQA Project

Code for paper TreeQA.

## Features

*   Processes complex questions by decomposing them into a logical tree structure.
*   Supports multiple standard QA datasets (WebQSP, QALD-EN, 2WikiMultihopQA, AdvHotpotQA, Musique).
*   Configurable components for Entity Linking (Azure, Relik), Embedding (NVIDIA, OpenAI), and Language Models (DeepSeek, Qwen, GPT).
*   Multithreaded dataset processing for faster inference.
*   Generates detailed JSONL output including reasoning steps, timings, and token counts.
*   Includes an evaluation script to calculate Exact Match (EM) scores and average resource consumption metrics.
*   Supports using alias lists (e.g., for WebQSP) during evaluation for more robust EM calculation.

## Prerequisites

*   Python (>= 3.10 recommended)

## Installation / Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url> # Replace with your actual repository URL
    cd TreeQA
    ```

2.  **Install Dependencies:**
    Install the required Python packages using the provided `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```


## Configuration (`treeQA_Config.py` Explained)

This file contains settings for various components of the TreeQA system.

**1. Entity Linking (EL)**

Determines how entities are identified and linked in the text. Choose *one* primary method.

*   `el_model`: Specifies the entity linking model to use.
    *   Set to `'relik'` to use the Relik service.
    *   Set to `'azure'` (or potentially another value if you integrate Azure EL directly in your code) to use Azure Cognitive Services Entity Linking.
*   **Azure Entity Linking Settings (if using Azure):**
    *   `Azure_key`: Your API key for the Azure Language Service. Get this from your Azure portal. (See: [Azure EL Overview](https://learn.microsoft.com/en-us/azure/ai-services/language-service/entity-linking/overview))
    *   `Azure_endpoint`: The endpoint URL for your Azure Language Service resource (e.g., `https://<your-resource-name>.cognitiveservices.azure.com/`).
*   **Relik Entity Linking Settings (if using Relik):**
    *   `relik_server_url`: The URL of the running Relik API server (e.g., `http://<server-ip>:<port>/api/relik`). (See: [Relik GitHub](https://github.com/SapienzaNLP/relik))
*   **Proxy Settings (Optional):**
    *   `proxies`: If you need to use an HTTP/HTTPS proxy to access external services (like Azure, Relik, or LLM APIs), configure it here. Set to `None` or an empty dictionary `{}` if no proxy is needed.
        ```python
        proxies = {
            "http": "http://your.proxy.address:port",
            "https": "http://your.proxy.address:port",
        }
        ```

**2. Embedding Model**

Used for retrieving relevant text chunks (from Wikipedia articles).

*   `RetrieveModelName`: Specifies the embedding model used for retrieval.
    *   Set to `'nv-embed-v2'` to use an NVIDIA embedding model endpoint.
    *   Set to `'text-embedding-3-small'` (or similar) if using OpenAI's embedding API (requires corresponding code integration).
*   `nv_embed_v2_url`: The URL endpoint for the NVIDIA embedding service (if `RetrieveModelName` is set accordingly).

**3. Language Model (LLM)**

The core model used for generating sub-questions, hypotheses, and final answers within the Logic Tree.

*   `model_name`: Specifies the primary LLM to use.
    *   Supported values (examples): `'deepseekV3-chat'`, `'qwen2.5-instruct-14b'`, `'gpt3.5-turbo'`. (Ensure your code in `LLMs/models.py` or similar handles the selected model).
*   **API Keys (Provide keys ONLY for the models you intend to use):**
    *   `aliApiKey`: Your API key from Alibaba Cloud for using Qwen models (e.g., via Model Studio). (See: [Alibaba Cloud API Key](https://help.aliyun.com/en/model-studio/developer-reference/get-api-key))
    *   `openaiApiKey`: Your API key from OpenAI for using GPT models. (See: [OpenAI API Keys](https://platform.openai.com/account/api-keys))
    *   `deepseekApiKey`: Your API key from DeepSeek for using their models. (See: [DeepSeek Platform](https://platform.deepseek.com/))

**4. Vector Store (for Retrieval)**

Settings related to storing and retrieving text chunks based on embeddings.

*   `Chroma_store`: Set to `True` if you are using ChromaDB as a vector store. Set to `False` otherwise.
*   `PersistentClient_Path`: **(Required if `Chroma_store` is `True`)** The local directory path where the ChromaDB persistent data is stored.
*   `chroma_collection_name`: **(Required if `Chroma_store` is `True`)** The name of the collection within ChromaDB that holds the article embeddings. *Important:* This name should correspond to the embedding model used to create the collection (e.g., `wikipediaNV` might imply it was created using `nv-embed-v2`).

**5. Search and Retrieval Parameters**

Control the breadth and depth of search and retrieval operations.

*   `ELTop_k`: The maximum number of top-ranked entities to consider from the Entity Linking results during the graph search phase.
*   `RLTop_k`: The maximum number of top-ranked relations (associated with entities) to consider during the graph search phase.
*   `article_top_k`: The number of most relevant text chunks (articles/paragraphs) to retrieve and provide to the LLM during the self-adaptive reasoning steps.

## Usage

The project provides scripts for running inference (`inference.py`) and evaluating the results (`evaluate.py`).

### 1. Inference (`inference.py`)

This script runs the main QA process using the Logic Tree. It has two modes: `single` for one question and `dataset` for batch processing.

**a) Single Question Mode**

**Command:**
```bash
python inference.py single --question "Your question text here?"
```
**Parameters:**
*   `single`: Mode specifier.
*   `--question "<TEXT>"`: **(Required)** The question text.

**b) Dataset Processing Mode**

**Command:**
```bash
python inference.py dataset --dataset_name <DATASET_NAME> --output_filename <OUTPUT_FILENAME.jsonl>
```
**Parameters:**
*   `dataset`: Mode specifier.
*   `--dataset_name <DATASET_NAME>`: **(Required)** Name of the dataset (e.g., `webqsp`, `qald-en`). Must match keys in `SUPPORTED_DATASETS`.
*   `--output_filename <OUTPUT_FILENAME.jsonl>`: **(Required)** Name for the output JSONL file (saved in `result/`).

### 2. Evaluation (`evaluate.py`)

Evaluates a JSONL result file, calculating EM (containment) and average metrics.

**Command:**
```bash
python evaluate.py <INPUT_FILE.jsonl> [--alias_file <ALIAS_FILE.json>] [--error_file <ERROR_FILE.jsonl>]
```
**Parameters:**
*   `<INPUT_FILE.jsonl>`: **(Required)** Path to the inference results file.
*   `--alias_file <ALIAS_FILE.json>`: **(Optional)** Path to a JSON alias file. If provided and valid, aliases are used for EM calculation.
*   `--error_file <ERROR_FILE.jsonl>`: **(Optional)** Path to save records where EM=0.

