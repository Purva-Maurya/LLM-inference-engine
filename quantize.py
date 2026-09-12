import torch
from model import blocks

def quantize_tensor(W):
    """
    Per-tensor symmetric INT8 quantization.
    Returns: (quantized int8 tensor, scale factor)
    """
    scale = W.abs().max() / 127.0
    W_int8 = torch.round(W / scale).clamp(-127, 127).to(torch.int8)
    return W_int8, scale


def dequantize_tensor(W_int8, scale):
    return W_int8.float() * scale


if __name__ == "__main__":
    # Test on a small example first
    W = torch.randn(4, 4) * 2  # some fake weight values

    print("Original weights:\n", W)

    W_int8, scale = quantize_tensor(W)
    print("\nQuantized (int8):\n", W_int8)
    print("Scale factor:", scale.item())

    W_reconstructed = dequantize_tensor(W_int8, scale)
    print("\nReconstructed (dequantized):\n", W_reconstructed)

    error = (W - W_reconstructed).abs()
    print("\nMax reconstruction error:", error.max().item())
    print("Mean reconstruction error:", error.mean().item())


if __name__ == "__main__":
    # ... (keep your existing test code above this) ...

    # Now test on a real weight from the model
    real_weight = blocks[0]["mlp_fc_w"]  # shape [768, 3072]
    print(f"\n=== Real weight: mlp_fc_w, shape {real_weight.shape} ===")

    original_size_bytes = real_weight.numel() * 4  # fp32 = 4 bytes per value
    quantized_size_bytes = real_weight.numel() * 1  # int8 = 1 byte per value

    print(f"Original size (fp32): {original_size_bytes / 1024:.1f} KB")
    print(f"Quantized size (int8): {quantized_size_bytes / 1024:.1f} KB")
    print(f"Memory reduction: {original_size_bytes / quantized_size_bytes:.1f}x")

    W_int8, scale = quantize_tensor(real_weight)
    W_reconstructed = dequantize_tensor(W_int8, scale)

    error = (real_weight - W_reconstructed).abs()
    relative_error = error / (real_weight.abs() + 1e-8)

    print(f"\nMax absolute error: {error.max().item():.6f}")
    print(f"Mean absolute error: {error.mean().item():.6f}")
    print(f"Mean relative error: {relative_error.mean().item() * 100:.2f}%")

def quantize_model_weights(blocks):
    """Quantize every weight matrix in every block, return quantized blocks + scales."""
    quantized_blocks = []
    for block in blocks:
        q_block = {}
        for key, tensor in block.items():
            if tensor.dim() == 2:  # only quantize actual weight matrices, not biases
                W_int8, scale = quantize_tensor(tensor)
                q_block[key] = dequantize_tensor(W_int8, scale)  # store dequantized for easy drop-in use
            else:
                q_block[key] = tensor.clone()  # biases stay fp32, tiny anyway
        quantized_blocks.append(q_block)
    return quantized_blocks


if __name__ == "__main__":
    # ... keep everything above ...

    print("\n=== Full model quantization: generation comparison ===")
    from model import tokenizer, wte, wpe, ln_f_w, ln_f_b, layer_norm, transformer_block, greedy_generate

    quantized_blocks = quantize_model_weights(blocks)

    # Temporarily swap in quantized weights and generate
    import model
    original_blocks = model.blocks
    model.blocks = quantized_blocks

    quantized_output = greedy_generate("The future of artificial intelligence is", max_new_tokens=30)

    model.blocks = original_blocks  # restore original weights
    original_output = greedy_generate("The future of artificial intelligence is", max_new_tokens=30)

    print("Original (fp32):  ", original_output)
    print("Quantized (int8):  ", quantized_output)
    print("Outputs match:", original_output == quantized_output) 

def quantize_tensor_per_channel(W, dim=0):
    """
    Per-channel symmetric INT8 quantization.
    Computes a separate scale for each slice along `dim`.
    """
    scale = W.abs().amax(dim=1 - dim, keepdim=True) / 127.0
    scale = scale.clamp(min=1e-8)  # avoid divide-by-zero on all-zero channels
    W_int8 = torch.round(W / scale).clamp(-127, 127).to(torch.int8)
    return W_int8, scale


def dequantize_tensor_per_channel(W_int8, scale):
    return W_int8.float() * scale


def quantize_model_weights_per_channel(blocks):
    quantized_blocks = []
    for block in blocks:
        q_block = {}
        for key, tensor in block.items():
            if tensor.dim() == 2:
                W_int8, scale = quantize_tensor_per_channel(tensor, dim=0)
                q_block[key] = dequantize_tensor_per_channel(W_int8, scale)
            else:
                q_block[key] = tensor.clone()
        quantized_blocks.append(q_block)
    return quantized_blocks


if __name__ == "__main__":
    # ... keep everything above ...

    print("\n=== Per-channel quantization: generation comparison ===")
    import model

    quantized_blocks_pc = quantize_model_weights_per_channel(model.blocks)
    original_blocks = model.blocks

    model.blocks = quantized_blocks_pc
    per_channel_output = greedy_generate("The future of artificial intelligence is", max_new_tokens=30)
    model.blocks = original_blocks

    print("Original (fp32):        ", original_output)
    print("Per-tensor quantized:    ", quantized_output)
    print("Per-channel quantized:   ", per_channel_output)       