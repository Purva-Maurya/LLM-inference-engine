import torch
from model import tokenizer

class Request:
    """Tracks one generation request's state as it moves through the system."""
    def __init__(self, request_id, prompt, max_new_tokens=20):
        self.request_id = request_id
        self.input_ids = tokenizer.encode(prompt, return_tensors="pt")[0].tolist()
        self.max_new_tokens = max_new_tokens
        self.num_generated = 0
        self.finished = False

    def is_done(self):
        if self.finished:
            return True
        if self.num_generated >= self.max_new_tokens:
            return True
        if len(self.input_ids) > 0 and self.input_ids[-1] == tokenizer.eos_token_id:
            return True
        return False


if __name__ == "__main__":
    requests = [
        Request(0, "Hello", max_new_tokens=10),
        Request(1, "The weather today is", max_new_tokens=15),
        Request(2, "I really enjoy programming because", max_new_tokens=20),
    ]
    for r in requests:
        print(f"Request {r.request_id}: tokens={r.input_ids}, done={r.is_done()}")

from model import wte, wpe, blocks, ln_f_w, ln_f_b, layer_norm
from batched_generate import mlp, transformer_block_batched


def run_scheduler(prompts, max_new_tokens_list, max_batch_size=2):
    tokenizer.pad_token = tokenizer.eos_token

    waiting = [Request(i, p, max_new_tokens_list[i]) for i, p in enumerate(prompts)]
    active = []
    finished = []

    step = 0
    while waiting or active:
        # Admit new requests into any free slots
        while len(active) < max_batch_size and waiting:
            active.append(waiting.pop(0))

        if not active:
            break

        # Build a padded batch from currently active requests
        seqs = [r.input_ids for r in active]
        max_len = max(len(s) for s in seqs)

        padded = []
        mask = []
        for s in seqs:
            pad_len = max_len - len(s)
            padded.append([tokenizer.eos_token_id] * pad_len + s)
            mask.append([0] * pad_len + [1] * len(s))

        input_ids = torch.tensor(padded)
        attn_mask = torch.tensor(mask)

        # One forward pass for the whole active batch
        position_ids = attn_mask.cumsum(dim=-1) - 1
        position_ids = position_ids.clamp(min=0)
        x = wte[input_ids] + wpe[position_ids]
        for block in blocks:
            x = transformer_block_batched(x, block, attn_mask)
        x = layer_norm(x, ln_f_w, ln_f_b)
        logits = x @ wte.T

        next_ids = torch.argmax(logits[:, -1, :], dim=-1)

        # Update each active request with its new token
        for i, r in enumerate(active):
            token = next_ids[i].item()
            r.input_ids.append(token)
            r.num_generated += 1

        # Evict finished requests
        still_active = []
        for r in active:
            if r.is_done():
                finished.append(r)
                print(f"[step {step}] Request {r.request_id} finished ({r.num_generated} tokens generated)")
            else:
                still_active.append(r)
        active = still_active

        step += 1

    finished.sort(key=lambda r: r.request_id)
    return [tokenizer.decode(r.input_ids, skip_special_tokens=True) for r in finished]


if __name__ == "__main__":
    prompts = ["Hello", "The weather today is", "I really enjoy programming because"]
    max_tokens = [5, 15, 8]  # deliberately different lengths to see eviction/admission happen

    outputs = run_scheduler(prompts, max_tokens, max_batch_size=2)
    for p, o in zip(prompts, outputs):
        print(f"\nPrompt: {p!r}\nOutput: {o!r}")        