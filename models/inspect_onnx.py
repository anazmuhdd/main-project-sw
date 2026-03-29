import onnx
import onnxruntime as ort
import numpy as np
import sys
import onnx.shape_inference

def main(model_path):
    print(f"Inspecting {model_path}...")
    onnx_proto = onnx.load(model_path)
    sess = ort.InferenceSession(model_path)
    
    # Check outputs
    print("\nOutputs:")
    for b in sess.get_outputs():
        print(f"  Name: {b.name}, Shape: {b.shape}")
    
    # Map from output name to node
    output_to_node = {}
    for node in onnx_proto.graph.node:
        for out in node.output:
            output_to_node[out] = node
            
    # To get intermediate shapes, we'll use onnx.shape_inference
    inferred_model = onnx.shape_inference.infer_shapes(onnx_proto)
    
    target_shape = [1, 84, 8400]
    target_shapes_alt = [1, 8400, 84]
    
    print("\nSearching for potential 'Raw' output nodes...")
    found = False
    for v in inferred_model.graph.value_info:
        shape = [d.dim_value for d in v.type.tensor_type.shape.dim]
        # Check if shape has 8400 and 84
        if (len(shape) == 3 and 
            ((shape[1] == 84 and shape[2] == 8400) or
             (shape[1] == 8400 and shape[2] == 84))):
            prod_node = output_to_node.get(v.name)
            print(f"  Found potential raw output: {v.name} with shape {shape} (produced by {prod_node.op_type if prod_node else 'unknown'})")
            found = True
            
    if not found:
        print("  None found with exactly [1, 84, 8400] or [1, 8400, 84]. Listing all nodes with 8400 in their shape.")
        for v in inferred_model.graph.value_info:
            shape = [d.dim_value for d in v.type.tensor_type.shape.dim]
            if 8400 in shape:
                prod_node = output_to_node.get(v.name)
                print(f"  {v.name} shape {shape} ({prod_node.op_type if prod_node else 'unknown'})")

if __name__ == "__main__":
    main(sys.argv[1])
