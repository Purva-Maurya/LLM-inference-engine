import torch 

def sample_with_temperature(logits, temperature=1.0):
    scaled_logits = logits / temperature
    probs = torch.softmax(scaled_logits, dim=-1)
    next_id = torch.multinomial(probs, num_samples=1).item()
    return next_id

def sample_with_temperature(logits, temperature=1.0):
    scaled_logits = logits / temperature
    probs = torch.softmax(scaled_logits, dim=-1)
    next_id = torch.multinomial(probs, num_samples=1).item()
    return next_id

def top_k_filter(logits, k):
    values, indices = torch.topk(logits, k)
    filtered_logits = torch.full_like(logits, float('-inf'))
    filtered_logits.scatter_(dim=-1, index=indices, src=values)
    return filtered_logits

def sample_with_temperature_and_topk(logits, temperature=1.0, k=10):
    filtered_logits = top_k_filter(logits, k)
    scaled_logits = filtered_logits / temperature
    probs = torch.softmax(scaled_logits, dim=-1)
    next_id = torch.multinomial(probs, num_samples=1).item()
    return next_id

def top_p_filter(logits, p=0.9):
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_idices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    sorted_mask = cumulative_probs > p
    sorted_mask[1:] =sorted_mask[ :-1].clone()
    sorted_mask[0] = False

    sorted_logits = torch.log(sorted_probs)
    sorted_logits[sorted_mask] = float('-inf')

    filtered_logits = torch.full_like(logits, float("-inf"))
    filtered_logits.scatter_(dim=-1, index=sorted_idices, src=sorted_logits)
    return filtered_logits


def sample_with_temperature_and_topp(logits, temperature=1.0, p=0.9):
    filtered_logits = top_p_filter(logits, p)
    scaled_logits = filtered_logits / temperature
    probs = torch.softmax(scaled_logits, dim=-1)
    next_id = torch.multinomial(probs, num_samples=1).item()
    return next_id

def sample_generate(forward_fn, tokenizer, prompt, max_new_tokens=30, temperature=1.0, top_k=None, top_p=None):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    for _ in range(max_new_tokens):
        logits = forward_fn(input_ids)
        last_logits = logits[0, -1, :]

        if top_p is not None:
            next_id = sample_with_temperature_and_topp(last_logits, temperature, top_p)
        elif top_k is not None:
            next_id = sample_with_temperature_and_topk(last_logits, temperature, top_k)
        else:
            next_id = sample_with_temperature(last_logits, temperature)

        next_id_tensor = torch.tensor([[next_id]])
        input_ids = torch.cat([input_ids, next_id_tensor], dim=1)

        if next_id == tokenizer.eos_token_id:
            break

    return tokenizer.decode(input_ids[0])