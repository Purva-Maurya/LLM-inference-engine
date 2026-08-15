from model import forward, tokenizer
from sampling import sample_with_temperature

input_ids = tokenizer.encode("The weather today is", return_tensors="pt")
logits = forward(input_ids)
last_logits = logits[0, -1, :]

print("=== Temperature 1.0 (neutral) ===")
for _ in range(5):
    tid = sample_with_temperature(last_logits, temperature=1.0)
    print(repr(tokenizer.decode([tid])))

print("\n=== Temperature 0.5 (more deterministic) ===")
for _ in range(5):
    tid = sample_with_temperature(last_logits, temperature=0.5)
    print(repr(tokenizer.decode([tid])))

print("\n=== Temperature 1.5 (more random) ===")
for _ in range(5):
    tid = sample_with_temperature(last_logits, temperature=1.5)
    print(repr(tokenizer.decode([tid])))

from sampling import sample_with_temperature_and_topk
print("\n=== Temperature 1.0 with Top-K (k=5) ===")
for _ in range(5):
    tid = sample_with_temperature_and_topk(last_logits, temperature=1.0, k=5)
    print(repr(tokenizer.decode([tid])))
print("\n=== Temperature 1.0 with Top-K (k=50) ===")
for _ in range(5):
    tid = sample_with_temperature_and_topk(last_logits, temperature=1.0, k=50)
    print(repr(tokenizer.decode([tid])))        


from sampling import sample_with_temperature_and_topp
print("\n=== Temperature 1.0 with Top-P (p=0.9) ===")
for _ in range(5):
    tid = sample_with_temperature_and_topp(last_logits, temperature=1.0, p=0.9)
    print(repr(tokenizer.decode([tid])))

print("\n=== Temperature 1.0 with Top-P (p=0.5) ===")
for _ in range(5):
    tid = sample_with_temperature_and_topp(last_logits, temperature=1.0, p=0.5)
    print(repr(tokenizer.decode([tid])))        

from sampling import sample_generate
print("\n=== Sample Generate with Temperature 1.0, Top-K=10 ===")
output = sample_generate(forward, tokenizer, "The future of AI is", max_new_tokens=30, temperature=0.8, top_p=0.9)

print(output)
