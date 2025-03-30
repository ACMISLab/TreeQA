import argparse
import concurrent
import json
import os
import sys
import time

from tqdm import tqdm

from treeQA.tree_class.logicTree import LogicTree


def answerQuestion(query):
    """Processes a single question, tracking time and tokens for stages."""
    global _global_token_counter
    _global_token_counter = 0 # Reset counter for each question

    start_time_total = time.perf_counter()

    logic_init_time = 0
    self_adaptive_time = 0
    final_reasoning_time = 0
    logic_init_tokens = 0
    self_adaptive_tokens = 0
    final_reasoning_tokens = 0
    total_tokens = 0
    processed_answer_tree = None
    fix_count = -1
    logic_tree= None
    try:
        tokens_before_init=0
        # 1. Logic Tree Initialization
        start_time_init = time.perf_counter()
        json_data,token_count = LogicTree.logic_tree_init(query)
        logic_tree = LogicTree(json_data)
        end_time_init = time.perf_counter()
        logic_tree.tokenCount += token_count
        logic_init_time = end_time_init - start_time_init
        logic_init_tokens = logic_tree.tokenCount - tokens_before_init

        # 2. Check and Refine (Self-Adaptive)
        start_time_refine = time.perf_counter()
        logic_tree.check_and_refine()
        end_time_refine = time.perf_counter()
        self_adaptive_time = end_time_refine - start_time_refine
        self_adaptive_tokens = logic_tree.tokenCount - logic_init_tokens

        # 3. Update Final Answer (Final Reasoning)
        start_time_update = time.perf_counter()
        logic_tree.update_final_answer()
        end_time_update = time.perf_counter()

        final_reasoning_time = end_time_update - start_time_update
        final_reasoning_tokens = logic_tree.tokenCount - self_adaptive_tokens

        # Get final results
        processed_answer_tree, fix_count = logic_tree.to_json()
        total_tokens = logic_tree.tokenCount

    except Exception as e:
        print(f"\nError processing question '{query[:50]}...': {e}", file=sys.stderr)
        # Record error state, return partial metrics if available
        processed_answer_tree = {"error": str(e), "query": query, "status": "failed"}
        fix_count = -1

    # Prepare metrics dictionary
    metrics = {
        "final_answer":logic_tree.data['answer'] if logic_tree else None,
        "logic_init_time": logic_init_time,
        "self_adaptive_time": self_adaptive_time,
        "final_reasoning_time": final_reasoning_time,
        "logic_init_tokens": logic_init_tokens,
        "self_adaptive_tokens": self_adaptive_tokens,
        "final_reasoning_tokens": final_reasoning_tokens,
        "total_tokens": total_tokens,
        "total_processing_time": time.perf_counter() - start_time_total # Optional: add total time
    }

    return processed_answer_tree, fix_count, metrics


# --- Dataset config and paths (Keep as before) ---
SUPPORTED_DATASETS = ["2wiki", "webqsp", "advhotpotqa", "qald-en", "musique"]
DATASET_FILE_MAP = {
    "2wiki": "dataset/2wikiMultihopQA/dev_sampled.json",
    "webqsp": "dataset/webqsp/WebQSP.json",
    "advhotpotqa": "dataset/advhotpot/hotpotadv_dev.json",
    "qald-en": "dataset/qald_10_en/qald_10-en.json",
    "musique": "dataset/musique/sampled_musique.json",
}
NUM_THREADS = 5
OUTPUT_DIR = "result"
# --- End Dataset config ---

def load_processed_ids(output_path):
    """Loads IDs of already processed questions from the output JSONL file."""
    processed_ids = set()
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        if 'id' in data:
                            processed_ids.add(data['id'])
                    except json.JSONDecodeError:
                        print(f"Warning: Skipping invalid JSON line in {output_path}: {line.strip()}", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Could not read processed IDs from {output_path}. Error: {e}", file=sys.stderr)
    print(f"Found {len(processed_ids)} already processed items in {output_path}")
    return processed_ids

def extract_data(item, dataset_name):
    """Extracts id, question, and original answer based on dataset format."""
    # --- (Keep the existing extract_data function as it was) ---
    try:
        if dataset_name == "2wiki":
            return item.get('_id'), item.get('question'), item.get('answer')
        elif dataset_name == "webqsp":
             q_id = item.get('QuestionId')
             question = item.get('ProcessedQuestion', item.get('RawQuestion'))
             answers = item.get('Parses', [{}])[0].get('Answers', [])
             original_answer = [ans.get('EntityName', ans.get('AnswerArgument')) for ans in answers]
             original_answer = [ans for ans in original_answer if ans]
             return q_id, question, original_answer
        elif dataset_name == "advhotpotqa":
            return item.get('qas_id'), item.get('question'), item.get('answer')
        elif dataset_name == "qald-en":

            q_id = f"qald_index_{item.get('__index__', 'unknown')}"
            # Question Extraction (handle list or string format, prefer English)
            question_data = item.get('question', [])  # Default to empty list
            question = ""
            if isinstance(question_data, list):
                for q_entry in question_data:
                    if isinstance(q_entry, dict) and q_entry.get('language') == 'en':
                        question = q_entry.get('string')
                        break
            elif isinstance(question_data, str):  # Handle case where question is just a string
                question = question_data
            # Add a check if question is still empty
            if not question:
                print(f"Warning: Could not extract English question for QALD item ID {q_id}", file=sys.stderr)

            # --- CORRECTED Answer Extraction for QALD ---
            answer_dict = item.get('answer', {})  # Get the answer dictionary, default to {}
            original_answer = []
            if isinstance(answer_dict, dict):
                original_answer = list(answer_dict.values())
            else:
                print(
                    f"Warning: Unexpected format for 'answer' field in QALD item ID {q_id}. Expected dict, got {type(answer_dict)}.",
                    file=sys.stderr)
            original_answer = list(set(filter(None, original_answer)))

            return q_id, question, original_answer
        elif dataset_name == "musique":
             ans = item.get('answer', '')
             ans_aliases = item.get('answer_aliases', [])
             original_answer = [ans] + ans_aliases if ans else ans_aliases
             return item.get('id'), item.get('question'), list(set(filter(None, original_answer)))
        else:
            raise ValueError(f"Unknown dataset format logic: {dataset_name}")
    except Exception as e:
        print(f"\nError extracting data for dataset {dataset_name} from item snippet: {str(item)[:200]}... Error: {e}", file=sys.stderr)
        return None, None, None
    # --- (End of extract_data) ---


# Helper function for multithreading - updated return values
def process_item_task(item_id, question_text, original_answer):
    """Task executed by each thread: processes one question and returns metrics."""
    processed_answer_tree, fix_count, metrics = answerQuestion(question_text)
    return item_id, question_text, original_answer, processed_answer_tree, fix_count, metrics


def process_dataset(dataset_name, dataset_file_path, output_file_path):
    """Loads, processes (multithreaded), and saves results including metrics."""
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    processed_ids = load_processed_ids(output_file_path)

    try:
        with open(dataset_file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            if dataset_name == "webqsp" and isinstance(raw_data, dict) and "Questions" in raw_data:
                 data = raw_data["Questions"]
            elif isinstance(raw_data, list):
                 data = raw_data
            else:
                 print(f"Error: Expected list/dict structure in {dataset_file_path}, got {type(raw_data)}", file=sys.stderr); sys.exit(1)
            if not isinstance(data, list):
                 print(f"Error: Could not extract list of questions from {dataset_file_path}", file=sys.stderr); sys.exit(1)
    except Exception as e:
        print(f"Error loading/parsing dataset {dataset_file_path}: {e}", file=sys.stderr); sys.exit(1)

    print(f"Processing dataset '{dataset_name}' from '{dataset_file_path}'...")
    print(f"Results will be saved to '{output_file_path}'")
    print(f"Using {NUM_THREADS} threads.")

    items_to_process_args = []
    skipped_count = 0
    processed_in_this_run = set()

    for i, item in enumerate(data):
        if not isinstance(item, dict): skipped_count += 1; continue
        item['__index__'] = i
        item_id, question_text, original_answer = extract_data(item, dataset_name)
        if item_id is None or question_text is None: skipped_count += 1; continue
        if item_id in processed_ids or item_id in processed_in_this_run: skipped_count += 1; continue
        items_to_process_args.append((item_id, question_text, original_answer))
        processed_in_this_run.add(item_id)

    if skipped_count > 0: print(f"Skipped {skipped_count} items.")
    if not items_to_process_args: print("No new items to process."); return

    with open(output_file_path, 'a', encoding='utf-8') as outfile:
        with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
            futures = [executor.submit(process_item_task, *args) for args in items_to_process_args]
            print(f"Submitting {len(futures)} questions for processing...")
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Processing {dataset_name}", unit="question"):
                try:
                    # Unpack the results including metrics
                    item_id, q_text, orig_ans, processed_ans_tree, fix_cnt, metrics = future.result()

                    # Prepare result dictionary, merging metrics
                    result = {
                        "id": item_id,
                        "question": q_text,
                        "original_answer": orig_ans,
                        "processed_answer": processed_ans_tree,
                        "fix_count": fix_cnt,
                        **metrics # Unpack the metrics dictionary into the result
                    }
                    outfile.write(json.dumps(result, ensure_ascii=False) + '\n')
                    outfile.flush()

                except Exception as exc:
                    # Log errors from the future execution itself
                    print(f'\nError retrieving result from thread: {exc}', file=sys.stderr)
                    # Optionally write an error marker to the output file
                    # error_result = {"id": "unknown", "error": str(exc), "status": "future_error", **{k: -1 for k in metrics.keys()}} # Add placeholder metrics
                    # outfile.write(json.dumps(error_result, ensure_ascii=False) + '\n')
                    # outfile.flush()


    print(f"\nFinished processing {dataset_name}. Results appended to {output_file_path}")


def process_single_question(question):
    """Processes a single question and prints the result with metrics."""
    print(f"Processing single question: \"{question}\"")
    processed_answer_tree, fix_count, metrics = answerQuestion(question) # Get metrics

    print("\n--- Processing Result ---")
    if isinstance(processed_answer_tree, dict) and processed_answer_tree.get("status") == "failed":
        print(f"Error: {processed_answer_tree.get('error')}")
    else:
        print(f"Fix Count: {fix_count}")
        print("Processed Answer (Tree):")
        print(processed_answer_tree)
    print(f"Final Answer is:        {metrics['final_answer']}")
    # Print metrics
    print("\n------------ Metrics ------------------")
    print(f"Logic Init Time:        {metrics['logic_init_time']:.4f}s")
    print(f"Self-Adaptive Time:     {metrics['self_adaptive_time']:.4f}s")
    print(f"Final Reasoning Time:   {metrics['final_reasoning_time']:.4f}s")
    print(f"Total Processing Time:  {metrics['total_processing_time']:.4f}s") # Optional total time
    print(f"Logic Init Tokens:      {metrics['logic_init_tokens']}")
    print(f"Self-Adaptive Tokens:   {metrics['self_adaptive_tokens']}")
    print(f"Final Reasoning Tokens: {metrics['final_reasoning_tokens']}")
    print(f"Total Tokens Consumed:  {metrics['total_tokens']}")
    print("------------------------------------------")


def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Process QA datasets or single questions using LogicTree with metrics.")
    subparsers = parser.add_subparsers(dest='mode', required=True, help='Operating mode: "dataset" or "single"')

    # --- Dataset Mode ---
    parser_dataset = subparsers.add_parser('dataset', help='Process a full dataset using multiple threads.')
    parser_dataset.add_argument('--dataset_name', choices=SUPPORTED_DATASETS, required=True,
                                help=f'Name of the dataset. Supported: {", ".join(SUPPORTED_DATASETS)}')
    parser_dataset.add_argument('--output_filename', type=str, required=True,
                                help=f'Output JSONL filename (e.g., results.jsonl). Saved in "{OUTPUT_DIR}/".')

    # --- Single Question Mode ---
    parser_single = subparsers.add_parser('single', help='Process a single question.')
    parser_single.add_argument('--question', type=str, required=True, help='The question text.')

    args = parser.parse_args()

    if args.mode == 'dataset':
        dataset_key = args.dataset_name
        if dataset_key not in DATASET_FILE_MAP:
            print(f"Error: Path undefined for dataset '{dataset_key}'.", file=sys.stderr); sys.exit(1)

        relative_dataset_path = DATASET_FILE_MAP[dataset_key]
        dataset_file_path = os.path.join(project_root, relative_dataset_path)
        output_filename = args.output_filename
        output_dir_path = os.path.join(project_root, OUTPUT_DIR)
        output_file_path = os.path.join(output_dir_path, output_filename)

        if not os.path.isfile(dataset_file_path):
             print(f"Error: Dataset file not found: '{dataset_file_path}'", file=sys.stderr); sys.exit(1)
        try:
            os.makedirs(output_dir_path, exist_ok=True)
        except OSError as e:
            print(f"Error creating output directory '{output_dir_path}': {e}", file=sys.stderr); sys.exit(1)

        process_dataset(args.dataset_name, dataset_file_path, output_file_path)

    elif args.mode == 'single':
        process_single_question(args.question)
    else:
        parser.print_help(); sys.exit(1)

if __name__ == "__main__":
    main()