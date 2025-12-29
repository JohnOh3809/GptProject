"""Test script to load and test the trained GPT model"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
import time

# Set seeds for reproducibility
np.random.seed(2025)
torch.manual_seed(2025)

print("=" * 80)
print("Loading Model and Testing Generation")
print("=" * 80)

# ============================================================================
# Model Architecture (copied from notebook)
# ============================================================================

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None, past_kv=None):
        batch_size = query.size(0)
        seq_len_q = query.size(1)
        seq_len_k_new = key.size(1)
        Q_proj = self.W_Q(query)
        K_proj = self.W_K(key)
        V_proj = self.W_V(value)
        Q = Q_proj.view(batch_size, seq_len_q, self.num_heads, self.d_k).transpose(1, 2)
        K = K_proj.view(batch_size, seq_len_k_new, self.num_heads, self.d_k).transpose(1, 2)
        V = V_proj.view(batch_size, seq_len_k_new, self.num_heads, self.d_k).transpose(1, 2)
        if past_kv is not None:
            K_past, V_past = past_kv
            K = torch.cat([K_past, K], dim=2)
            V = torch.cat([V_past, V], dim=2)
        present_kv = (K, V)
        seq_len_k_full = K.size(2)
        scores = (Q @ K.transpose(-2, -1)) / (self.d_k ** 0.5)
        if mask is not None:
            scores = scores + mask.unsqueeze(0).unsqueeze(0)
        weights = torch.softmax(scores, dim=-1)
        output_attn = weights @ V
        output_attn = output_attn.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.d_model)
        output = self.W_O(output_attn)
        return output, weights, present_kv


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)

    def forward(self, x, past_kv=None):
        batch_size, seq_len, d_model = x.shape
        if past_kv is not None:
            mask = None
        else:
            mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)
            mask = mask.masked_fill(mask, float('-inf'))
        output, attn_weights, present_kv = self.mha(x, x, x, mask=mask, past_kv=past_kv)
        return output, attn_weights, present_kv


class MyLayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        mean = torch.mean(x, dim=-1, keepdim=True)
        var = torch.var(x, dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff=None):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.ln1 = MyLayerNorm(d_model)
        self.ln2 = MyLayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x, past_kv=None):
        attn_output, _, present_kv = self.attn(self.ln1(x), past_kv=past_kv)
        x = x + attn_output
        x = x + self.ffn(self.ln2(x))
        return x, present_kv


class DecoderLM(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, max_seq_len=512):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, num_heads) for _ in range(num_layers)])
        self.ln_f = MyLayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x, past_kv_list=None):
        batch_size, seq_len = x.shape
        tok_emb = self.token_embed(x)
        if past_kv_list is not None:
            past_seq_len = past_kv_list[0][0].shape[2] if past_kv_list else 0
            current_position_idx = past_seq_len
            positions_to_embed = torch.tensor([current_position_idx], device=x.device)
            pos_emb = self.pos_embed(positions_to_embed)
            pos_emb = pos_emb.unsqueeze(0)
        else:
            positions = torch.arange(seq_len, device=x.device)
            pos_emb = self.pos_embed(positions)
            pos_emb = pos_emb.unsqueeze(0)
        h = tok_emb + pos_emb
        present_kv_list = []
        for i, block in enumerate(self.blocks):
            past_kv_i = past_kv_list[i] if past_kv_list else None
            h, present_kv_i = block(h, past_kv=past_kv_i)
            present_kv_list.append(present_kv_i)
        h = self.ln_f(h)
        logits = self.head(h)
        return logits, present_kv_list


# ============================================================================
# Tokenizer (copied from notebook)
# ============================================================================

class BPETokenizer:
    def __init__(self):
        self.merges = []
        self.vocab = {"<UNK>": 0}
        self.id_to_token = {0: "<UNK>"}
        self.next_id = 1

    def _get_pairs(self, tokens):
        pairs = Counter()
        for i in range(len(tokens) - 1):
            pairs[(tokens[i], tokens[i+1])] += 1
        return pairs

    def _merge_pair(self, tokens, pair, new_token):
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i+1 < len(tokens) and (tokens[i], tokens[i+1]) == pair:
                new_tokens.append(new_token)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens

    def fit(self, text, num_merges=10):
        self.merges = []
        self.vocab = {"<UNK>": 0}
        self.id_to_token = {0: "<UNK>"}
        self.next_id = 1
        initial_chars = sorted(list(set(text)))
        for char in initial_chars:
            if char not in self.vocab:
                self.vocab[char] = self.next_id
                self.id_to_token[self.next_id] = char
                self.next_id += 1
        current_tokens = list(text)
        for _ in range(num_merges):
            pairs = self._get_pairs(current_tokens)
            if not pairs:
                break
            most_frequent_pair = max(pairs, key=pairs.get)
            new_token_str = "".join(most_frequent_pair)
            if new_token_str not in self.vocab:
                self.vocab[new_token_str] = self.next_id
                self.id_to_token[self.next_id] = new_token_str
                self.next_id += 1
                self.merges.append((most_frequent_pair, new_token_str))
            else:
                if (most_frequent_pair, new_token_str) not in self.merges:
                    self.merges.append((most_frequent_pair, new_token_str))
            current_tokens = self._merge_pair(current_tokens, most_frequent_pair, new_token_str)

    def encode(self, text):
        initial_tokens_list = []
        for char in text:
            if char not in self.vocab:
                self.vocab[char] = self.next_id
                self.id_to_token[self.next_id] = char
                self.next_id += 1
            initial_tokens_list.append(char)
        tokens_to_merge = list(initial_tokens_list)
        for pair, new_token_str in self.merges:
            tokens_to_merge = self._merge_pair(tokens_to_merge, pair, new_token_str)
        encoded_ids = [self.vocab.get(token, self.vocab["<UNK>"]) for token in tokens_to_merge]
        return encoded_ids

    def decode(self, ids):
        decoded_tokens = [self.id_to_token.get(id, "<UNK>") for id in ids]
        return "".join(decoded_tokens)


# ============================================================================
# Generation Function
# ============================================================================

def generate_text(model, tokenizer, prompt, max_new_tokens=200, temperature=1.0, top_k=None, top_p=None, use_kv_cache=True):
    """Generate text using the model with KV caching support."""
    model.eval()
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    generated_ids = input_ids.copy()
    past_kv_list = None
    
    with torch.no_grad():
        if use_kv_cache:
            logits, past_kv_list = model(input_tensor, past_kv_list=None)
            next_token_logits = logits[0, -1, :] / temperature
        else:
            logits, _ = model(input_tensor, past_kv_list=None)
            next_token_logits = logits[0, -1, :] / temperature
        
        for _ in range(max_new_tokens):
            if top_k is not None:
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                filtered_logits = torch.full_like(next_token_logits, float('-inf'))
                filtered_logits[top_k_indices] = top_k_logits
                next_token_logits = filtered_logits
            
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float('-inf')
            
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1).item()
            generated_ids.append(next_token_id)
            
            if use_kv_cache:
                next_input = torch.tensor([[next_token_id]], dtype=torch.long).to(device)
                logits, past_kv_list = model(next_input, past_kv_list=past_kv_list)
                next_token_logits = logits[0, -1, :] / temperature
            else:
                input_tensor = torch.tensor([generated_ids], dtype=torch.long).to(device)
                logits, _ = model(input_tensor, past_kv_list=None)
                next_token_logits = logits[0, -1, :] / temperature
    
    generated_text = tokenizer.decode(generated_ids)
    return generated_text


# ============================================================================
# Main: Load and Test
# ============================================================================

print("\n1. Loading tokenizer...")
with open('data/input.txt', 'r') as f:
    text_data = f.read()

tokenizer = BPETokenizer()
tokenizer.fit(text_data, num_merges=1000)
print(f"   Tokenizer fitted. Vocabulary size: {len(tokenizer.vocab)}")

print("\n2. Loading model checkpoint...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"   Using device: {device}")

# Model hyperparameters (from training)
vocab_size = len(tokenizer.vocab)
d_model = 256
num_heads = 8
num_layers = 6
max_seq_len = 256

model = DecoderLM(vocab_size, d_model, num_heads, num_layers, max_seq_len=max_seq_len)
model.to(device)

checkpoint = torch.load('best_model_checkpoint.pt', map_location=device)
model.load_state_dict(checkpoint)
model.eval()
print("   Model loaded successfully!")

print("\n3. Testing text generation...")
print("=" * 80)

test_prompts = [
    "To be or not to be",
    "Once upon a time",
    "The king said",
    "Romeo and Juliet",
    "All the world's a stage"
]

for i, prompt in enumerate(test_prompts, 1):
    print(f"\nTest {i}: Prompt = '{prompt}'")
    print("-" * 80)
    try:
        generated = generate_text(
            model, 
            tokenizer, 
            prompt, 
            max_new_tokens=150,
            temperature=0.8,
            use_kv_cache=True
        )
        print(f"Generated text:\n{generated}\n")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

print("\n4. Testing different temperatures...")
print("=" * 80)
prompt = "To be or not to be"
for temp in [0.3, 0.7, 1.0, 1.5]:
    print(f"\nTemperature = {temp}")
    print("-" * 80)
    try:
        generated = generate_text(
            model, tokenizer, prompt,
            max_new_tokens=100, temperature=temp, use_kv_cache=True
        )
        print(f"{generated}\n")
    except Exception as e:
        print(f"Error: {e}")

print("\n5. Testing KV caching performance...")
print("=" * 80)
prompt = "The king said to his court"
num_tokens = 200

start_time = time.time()
generated_with_cache = generate_text(
    model, tokenizer, prompt,
    max_new_tokens=num_tokens, temperature=0.8, use_kv_cache=True
)
time_with_cache = time.time() - start_time

start_time = time.time()
generated_without_cache = generate_text(
    model, tokenizer, prompt,
    max_new_tokens=num_tokens, temperature=0.8, use_kv_cache=False
)
time_without_cache = time.time() - start_time

speedup = time_without_cache / time_with_cache
print(f"\nGeneration time with KV cache: {time_with_cache:.4f} seconds")
print(f"Generation time without KV cache: {time_without_cache:.4f} seconds")
print(f"Speedup: {speedup:.2f}x")
print(f"\nOutputs match: {generated_with_cache == generated_without_cache}")

print("\n" + "=" * 80)
print("Testing complete!")

# Save outputs to file
print("\n6. Saving outputs to 'generation_outputs.txt'...")
with open('generation_outputs.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("Model Generation Test Results\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("TEST 1: Various Prompts\n")
    f.write("=" * 80 + "\n\n")
    for i, prompt in enumerate(test_prompts, 1):
        f.write(f"Prompt {i}: '{prompt}'\n")
        f.write("-" * 80 + "\n")
        try:
            generated = generate_text(
                model, tokenizer, prompt,
                max_new_tokens=150, temperature=0.8, use_kv_cache=True
            )
            f.write(f"{generated}\n\n")
        except Exception as e:
            f.write(f"Error: {e}\n\n")
    
    f.write("\n" + "=" * 80 + "\n")
    f.write("TEST 2: Different Temperature Settings\n")
    f.write("=" * 80 + "\n\n")
    prompt = "To be or not to be"
    for temp in [0.3, 0.7, 1.0, 1.5]:
        f.write(f"Temperature = {temp}\n")
        f.write("-" * 80 + "\n")
        try:
            generated = generate_text(
                model, tokenizer, prompt,
                max_new_tokens=100, temperature=temp, use_kv_cache=True
            )
            f.write(f"{generated}\n\n")
        except Exception as e:
            f.write(f"Error: {e}\n\n")
    
    f.write("\n" + "=" * 80 + "\n")
    f.write("TEST 3: KV Caching Performance\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Generation time with KV cache: {time_with_cache:.4f} seconds\n")
    f.write(f"Generation time without KV cache: {time_without_cache:.4f} seconds\n")
    f.write(f"Speedup: {speedup:.2f}x\n")
    f.write(f"Outputs match: {generated_with_cache == generated_without_cache}\n\n")
    f.write("Generated with cache (first 500 chars):\n")
    f.write(generated_with_cache[:500] + "\n\n")
    f.write("Generated without cache (first 500 chars):\n")
    f.write(generated_without_cache[:500] + "\n")

print("   Outputs saved to 'generation_outputs.txt'")
print("\nOpening file...")

