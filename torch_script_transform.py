"""
Author: standard_lee@inha.edu
Date: 2024-11-25
Version: 1.1
Description: A script for transforming a .pth file to a TorchScript or ONNX runtime file.

Usage:


1. TorchScript: Convert the model to TorchScript format.
- python torch_script_transform.py --task torchscript --model-path model.pth --save-path LSNet_scripted.pt

2. Quantize: Apply dynamic quantization to the model.
- python torch_script_transform.py --task quantize --model-path model.pth --save-path LSNet_quantized.pt

3. ONNX: Convert the model to ONNX format.
- python torch_script_transform.py --task onnx --model-path model.pth --save-path LSNet.onnx


4. Simplify: Simplify the ONNX model using onnx-simplifier.
- python torch_script_transform.py --task simplify --onnx-path LSNet_optimized.onnx --save-path LSNet_simplified.onnx

5. Executorch: Convert the model for Executorch runtime.
- python torch_script_transform.py --task executorch --model-path model.pth --save-path LSNet_executorch.pt

!!! Disclaimer: CDD (ChatGPT Driven Development) WARNING !!! 
- Most of the code was generated from questions and answers using ChatGPT
"""

import argparse
import torch
from LSNet import LSNet
import onnx
from onnxruntime_tools import optimizer
from onnxsim import simplify
from torch.utils.mobile_optimizer import optimize_for_mobile

def load_model(model_path):
    """Load the LSNet model with the provided weights."""
    model = LSNet()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

def convert_to_torchscript(model, save_path):
    """Convert model to TorchScript format (CPU-only)."""
    example_rgb = torch.randn(1, 3, 224, 224)  # Example RGB input
    example_ti = torch.randn(1, 3, 224, 224)   # Example TI input
    scripted_model = torch.jit.trace(model, (example_rgb, example_ti))
    scripted_model.save(save_path)
    print(f"TorchScript model saved to {save_path}")

def quantize_model(model, save_path):
    """Apply dynamic quantization to the model (CPU-only)."""
    quantized_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    example_rgb = torch.randn(1, 3, 224, 224)  # Example RGB input
    example_ti = torch.randn(1, 3, 224, 224)   # Example TI input
    scripted_quantized_model = torch.jit.trace(quantized_model, (example_rgb, example_ti))
    scripted_quantized_model.save(save_path)
    print(f"Quantized TorchScript model saved to {save_path}")

def convert_to_onnx(model, save_path):
    """Convert model to ONNX format (CPU-only)."""
    example_rgb = torch.randn(1, 3, 224, 224)  # Example RGB input
    example_ti = torch.randn(1, 3, 224, 224)   # Example TI input

    torch.onnx.export(
        model,
        (example_rgb, example_ti),
        save_path,
        export_params=True,
        opset_version=11,
        input_names=["rgb", "ti"],  # Input tensor names
        output_names=["output"],    # Output tensor name
        dynamic_axes={
            "rgb": {0: "batch_size"}, 
            "ti": {0: "batch_size"}, 
            "output": {0: "batch_size"}
        }
    )
    print(f"ONNX model saved to {save_path}")

def simplify_onnx(onnx_path, simplified_path):
    """Simplify ONNX model using onnx-simplifier."""
    model = onnx.load(onnx_path)
    model_simplified, check = simplify(model)
    if check:
        onnx.save(model_simplified, simplified_path)
        print(f"Simplified ONNX model saved to {simplified_path}")
    else:
        print("Simplification failed!")

def convert_to_executorch(model, save_path):
    """Convert model to Executorch format (CPU-only)."""
    example_rgb = torch.randn(1, 3, 224, 224)  # Example RGB input
    example_ti = torch.randn(1, 3, 224, 224)   # Example TI input
    scripted_model = torch.jit.trace(model, (example_rgb, example_ti))
    
    # Optimize for mobile (Executorch runtime compatibility)
    optimized_model = optimize_for_mobile(scripted_model)
    optimized_model._save_for_lite_interpreter(save_path)
    print(f"Executorch model saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="LSNet Model Conversion and Optimization Tool")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the trained model (.pth)")
    parser.add_argument("--task", type=str, choices=["torchscript", "quantize", "onnx", "simplify", "executorch"],
                        required=True, help="Task to perform")
    parser.add_argument("--save-path", type=str, required=True, help="Path to save the output model")
    parser.add_argument("--onnx-path", type=str, help="Path to existing ONNX model (for simplify)")

    args = parser.parse_args()
    
    if args.task in ["torchscript", "quantize", "onnx", "executorch"]:
        model = load_model(args.model_path)  # Load model for CPU-only operations

    if args.task == "torchscript":
        convert_to_torchscript(model, args.save_path)
    elif args.task == "quantize":
        quantize_model(model, args.save_path)
    elif args.task == "onnx":
        convert_to_onnx(model, args.save_path)
    elif args.task == "simplify":
        if not args.onnx_path:
            raise ValueError("For 'simplify', you must provide --onnx-path")
        simplify_onnx(args.onnx_path, args.save_path)

if __name__ == "__main__":
    main()