import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from model import forward, tokenizer

def test_logits_match():
    prompt = "Hello, my name is"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Official HF forward pass
    hf_model = GPT2LMHeadModel.from_pretrained("gpt2")
    hf_model.eval()
    with torch.no_grad():
        hf_logits = hf_model(input_ids).logits

    # Our manual forward pass
    manual_logits = forward(input_ids)

    max_diff = (hf_logits - manual_logits).abs().max().item()
    print(f"Max absolute logit difference: {max_diff:.6f}")

    hf_preds = hf_logits.argmax(dim=-1)
    manual_preds = manual_logits.argmax(dim=-1)
    match_rate = (hf_preds == manual_preds).float().mean().item()
    print(f"Argmax prediction match rate: {match_rate * 100:.1f}%")

    assert max_diff < 1e-3, f"Logits diverge too much: {max_diff}"
    assert match_rate == 1.0, "Predicted tokens diverge from HF reference"
    print("\n✅ Manual forward pass matches HuggingFace reference.")


if __name__ == "__main__":
    test_logits_match()