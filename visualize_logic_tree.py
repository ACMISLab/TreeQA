import json
from graphviz import Digraph
import textwrap # To wrap long text in nodes
import itertools # For generating unique node IDs
import os # To ensure directory exists for output



def format_label(text, max_width=50, indent=''):
    """Wraps text for better display in nodes, with optional indent."""
    wrapped_lines = textwrap.wrap(text, width=max_width, subsequent_indent=indent + '  ')
    return indent + ('\n' + indent).join(wrapped_lines)

def add_nodes_edges(graph, node_data, parent_id, id_counter):
    """Recursively adds nodes and edges to the graph, including ref data."""
    for child in node_data.get('children', []):
        # Generate a unique ID for the current node
        current_id = f"node_{next(id_counter)}"

        # --- Prepare the label content, including refs ---
        label_parts = []
        sub_q = child.get("sub_question", "N/A")
        hyp_a = child.get("hypothesis_answer", "N/A")

        label_parts.append(f"Sub-Question:\n{format_label(sub_q, indent='  ')}")
        label_parts.append(f"\nHypothesis:\n{format_label(hyp_a, indent='  ')}")

        # Add Reference information if present
        ref_data = child.get("ref")
        if ref_data:
            label_parts.append("\n\nReferences:")
            wiki_refs = ref_data.get("wikipedia")
            if wiki_refs:
                label_parts.append("  Wikipedia:")
                for ref_text in wiki_refs:
                    # Split key/title from text if possible
                    parts = ref_text.split('||', 1)
                    if len(parts) == 2:
                        title, text = parts
                        label_parts.append(f"    - {title.strip()}:\n{format_label(text.strip(), indent='      ')}")
                    else: # Handle case where there's no '||'
                         label_parts.append(f"{format_label(ref_text.strip(), indent='    - ')}")


            wikidata_refs = ref_data.get("wikidata")
            if wikidata_refs:
                label_parts.append("\n  Wikidata:")
                for ref_text in wikidata_refs:
                    label_parts.append(f"{format_label(ref_text.strip(), indent='    - ')}")

        label_text = '\n'.join(label_parts)
        # --- End of label preparation ---

        # Add the node to the graph
        # Using align='left' within the node by setting label justification to left (\l)
        # Note: Graphviz uses \l, \r, \n for left, right, center justification within labels
        graph.node(current_id, label=label_text + r'\l', shape='box', style='filled, rounded', fillcolor='lightblue')

        # Add an edge from the parent to the current node
        graph.edge(parent_id, current_id)

        # Recursively add children of the current node
        if child.get('children'):
            add_nodes_edges(graph, child, current_id, id_counter)

def visualize_logic_tree(json_data_string, output_filename='logic_tree_full_visualization', output_dir='.', view=True):
    """
    Parses the JSON string and generates a tree visualization including all data.

    Args:
        json_data_string (str): The JSON string containing the tree data.
        output_filename (str): The base name for the output file (without extension).
        output_dir (str): The directory to save the output files.
        view (bool): Whether to attempt to open the generated image.
    """
    try:
        data = json.loads(json_data_string)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    output_path_base = os.path.join(output_dir, output_filename)

    # Initialize the graph
    # Increased size slightly, adjust as needed if text overflows
    dot = Digraph(comment='Logic Tree Full', format='png')
    dot.attr(rankdir='TB', size='15,15', overlap='false', splines='spline', nodesep='0.5', ranksep='0.8')
    dot.attr('node', shape='box', style='filled, rounded', fontname='Arial', fontsize='10', labelloc='l', nojustify='false') # labelloc/nojustify for left align
    dot.attr('edge', fontname='Arial', fontsize='9')

    # Unique ID generator
    id_counter = itertools.count()

    # Add the root node (Input Question & Final Answer)
    root_id = f"node_{next(id_counter)}"
    input_question = data.get("input_question", "Input Question Missing")
    final_answer = data.get("answer", "Final Answer Missing")
    root_label_parts = [
        f"Input Question:\n{format_label(input_question, indent='  ')}",
        f"\n\nFinal Answer:\n{format_label(final_answer, indent='  ')}"
    ]
    root_label = '\n'.join(root_label_parts)
    # Using Mrecord might offer more structure, but box with left-align often works well.
    # Added \l for left justification within the node.
    dot.node(root_id, label=root_label + r'\l', shape='box', style='filled', fillcolor='lightcoral')

    # Add the rest of the tree starting from the children of logic_tree
    if "logic_tree" in data and "children" in data["logic_tree"]:
        add_nodes_edges(dot, data["logic_tree"], root_id, id_counter)
    else:
        print("Warning: 'logic_tree' or its 'children' not found in JSON data.")

    # Render the graph
    try:
        # Render to file and automatically open it if view=True
        # cleanup=True removes the intermediate DOT source file
        rendered_path = dot.render(output_path_base, view=view, cleanup=True, format='png')
        print(f"Graph saved to {rendered_path}")
    except Exception as e:
        # Catch potential errors if Graphviz executable is not found
        print(f"Error rendering graph: {e}")
        print("Please ensure Graphviz is installed and its 'bin' directory is in your system's PATH.")
        print("Download from: https://graphviz.org/download/")
        # Save the DOT source file for manual rendering
        dot_source_path = f"{output_path_base}.gv"
        try:
            dot.save(dot_source_path)
            print(f"DOT source file saved to {dot_source_path}. You can render it manually using 'dot -Tpng \"{dot_source_path}\" -o \"{output_path_base}.png\"'")
        except Exception as save_e:
            print(f"Error saving DOT source file: {save_e}")


# --- Execute the visualization ---
# You can change the output directory and filename if needed
visualize_logic_tree(json_string, output_filename='logic_tree_full_visualization', output_dir='.', view=True)
