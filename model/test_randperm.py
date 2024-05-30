import torch

def sample_indices_from_tensors_vectorized(num_masked_tokens, num_tokens, B):
    random_matrix = torch.rand(B, num_tokens)
    sorted_indices = random_matrix.argsort(dim=1)
    sampled_indices = sorted_indices[:, :num_masked_tokens]
    return sampled_indices

num_masked_tokens = 9
num_tokens = 10
batch_size = 5
sampled_indices_tensor = sample_indices_from_tensors_vectorized(num_masked_tokens, num_tokens, batch_size)
print(sampled_indices_tensor)
