import onnx
import onnx_graphsurgeon as gs
import sys

def prune_yolo_head(input_path, output_path):
    print(f"Loading {input_path}...")
    # Load and infer shapes to ensure dtype/shape info is present
    model = onnx.load(input_path)
    try:
        model = onnx.shape_inference.infer_shapes(model)
    except:
        pass
        
    graph = gs.import_onnx(model)
    
    # We'll search for the TopK nodes and trace back to their inputs
    topk_nodes = [node for node in graph.nodes if node.op == "TopK"]
    if not topk_nodes:
        print("No TopK nodes found. Listing all nodes to find potential outputs.")
        # Fallback: Find nodes with large anchor counts
        return
    
    new_outputs = []
    seen_names = set()
    for node in topk_nodes:
        for inp in node.inputs:
            if isinstance(inp, gs.Variable) and inp.name not in seen_names:
                # Ensure dtype exists, default to float32 if missing (standard for YOLO heads)
                if inp.dtype is None:
                    inp.dtype = graph.inputs[0].dtype if graph.inputs else "float32"
                new_outputs.append(inp)
                seen_names.add(inp.name)
                print(f"Adding output: {inp.name} with shape {inp.shape} and dtype {inp.dtype}")
    
    graph.outputs = new_outputs
    graph.cleanup().toposort()
    
    print(f"Saving pruned model to {output_path}...")
    onnx.save(gs.export_onnx(graph), output_path)

if __name__ == "__main__":
    prune_yolo_head(sys.argv[1], sys.argv[2])
