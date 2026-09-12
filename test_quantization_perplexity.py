import torch
import torch.nn.functional as F
from model import tokenizer, forward
from quantize import quantize_model_weights, quantize_model_weights_per_channel
import model as model_module

# Longer, fixed evaluation corpus — several hundred words, held constant
# across all comparisons. Real, coherent English text (not model-generated).
EVAL_TEXT = """
The history of artificial intelligence began in antiquity, with myths and
stories of artificial beings endowed with intelligence by master craftsmen.
Modern research into the field started shortly after World War Two, when
scientists began to explore whether machines could be made to think.

Early pioneers believed that machine intelligence was achievable within a
generation, and it would take decades before researchers fully appreciated
the difficulty of the problem. Progress slowed considerably during periods
that came to be known as AI winters, when funding and interest declined
sharply after early optimism failed to produce practical results.

Despite these setbacks, steady advances continued in narrower subfields.
Researchers made progress on search algorithms, knowledge representation,
and eventually statistical methods that allowed systems to learn patterns
directly from data rather than relying solely on hand-crafted rules.

The rise of large datasets and increased computational power in the twenty
first century enabled a resurgence often called the deep learning era.
Neural networks, particularly those with many layers, began to achieve
results on tasks such as image recognition and language processing that
had previously seemed far out of reach for automated systems.

Transformers, introduced in the latter part of the twenty tens, further
accelerated progress in natural language processing. Their ability to
model relationships between distant words in a sequence made them
particularly well suited to tasks involving long passages of text, and
they quickly became the dominant architecture for language models.
""".strip()


def compute_perplexity(text):
    """Perplexity: exp(average negative log-likelihood over the sequence).
    Measures next-token prediction quality on this specific evaluation text —
    not a claim about general language understanding."""
    input_ids = tokenizer.encode(text, return_tensors="pt")

    with torch.no_grad():
        logits = forward(input_ids)

    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

    avg_neg_log_likelihood = -token_log_probs.mean()
    perplexity = torch.exp(avg_neg_log_likelihood).item()
    return perplexity, input_ids.shape[1]


def estimate_memory_mb(blocks):
    """Rough weight-only memory estimate: sum of all 2D weight tensor bytes."""
    total_bytes = 0
    for block in blocks:
        for key, tensor in block.items():
            if tensor.dim() == 2:
                total_bytes += tensor.numel() * 4  # fp32 storage size, since we store dequantized
    return total_bytes / (1024 ** 2)


def run_perplexity_comparison():
    original_blocks = model_module.blocks

    # --- fp32 baseline ---
    model_module.blocks = original_blocks
    ppl_fp32, num_tokens = compute_perplexity(EVAL_TEXT)
    mem_fp32 = estimate_memory_mb(original_blocks)

    # --- per-tensor quantized (independent copy) ---
    per_tensor_blocks = quantize_model_weights(original_blocks)
    model_module.blocks = per_tensor_blocks
    ppl_per_tensor, _ = compute_perplexity(EVAL_TEXT)

    # --- per-channel quantized (independent copy) ---
    per_channel_blocks = quantize_model_weights_per_channel(original_blocks)
    model_module.blocks = per_channel_blocks
    ppl_per_channel, _ = compute_perplexity(EVAL_TEXT)

    model_module.blocks = original_blocks  # restore before printing/exiting

    # --- Sanity check: confirm fp32 baseline wasn't mutated by the quantization calls ---
    ppl_fp32_recheck, _ = compute_perplexity(EVAL_TEXT)
    assert abs(ppl_fp32 - ppl_fp32_recheck) < 1e-6, \
        "fp32 baseline changed after quantization calls — possible in-place mutation bug!"

    print(f"Evaluation corpus: {num_tokens} tokens\n")
    print(f"{'Precision':<18}{'Perplexity':<14}{'Δ vs fp32':<14}{'Weight memory'}")
    print("-" * 60)
    print(f"{'fp32 (baseline)':<18}{ppl_fp32:<14.3f}{'—':<14}{mem_fp32:.1f} MB")
    delta_pt = (ppl_per_tensor / ppl_fp32 - 1) * 100
    print(f"{'INT8 per-tensor':<18}{ppl_per_tensor:<14.3f}{delta_pt:+.1f}%{'':<8}{mem_fp32/4:.1f} MB (int8 storage)")
    delta_pc = (ppl_per_channel / ppl_fp32 - 1) * 100
    print(f"{'INT8 per-channel':<18}{ppl_per_channel:<14.3f}{delta_pc:+.1f}%{'':<8}{mem_fp32/4:.1f} MB (int8 storage)")

    print(f"\n✅ fp32 baseline confirmed unmutated after quantization runs (sanity check passed).")


if __name__ == "__main__":
    run_perplexity_comparison()