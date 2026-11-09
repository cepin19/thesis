import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM

# Load the pre-trained model and tokenizer
#model_name = "distilroberta-base"
model_name = "bigscience/bloom"
model_name = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
#model = AutoModelForMaskedLM.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(model_name,device_map="auto",torch_dtype="auto",cache_dir="/home/enki/models/marian/perplex/models/",offload_folder="/home/enki/models/marian/perplex/models/")
# Input text
text = "I love machine learning."

# Tokenize the text and get tokens and attention_mask
inputs = tokenizer(text, return_tensors="pt")
tokens = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

# Get the logits from the model
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# Compute the probability distribution
probs = torch.softmax(logits, dim=-1)

# Get the actual probability of the true tokens
true_probs = torch.gather(probs, 2, tokens.unsqueeze(-1)).squeeze(-1)

# Compute the negative log likelihood
neg_log_likelihood = -torch.log(true_probs)

# Compute the perplexity per token (ignoring special tokens)
token_perplexities = neg_log_likelihood * attention_mask
token_perplexities[token_perplexities == 0] = float("nan")  # replace zeros with NaN

# Decode tokens to words
decoded_tokens = [tokenizer.decode(token) for token in tokens[0]]

# Print token-level perplexities
print("Token-level perplexities:")
for token, perplexity in zip(decoded_tokens, token_perplexities[0]):
    print(f"{token}: {perplexity.item()}")

