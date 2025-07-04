import torch
import matplotlib.pyplot as plt

# Create a sample matrix
m, n, k = 1024, 4096, 20  # matrix size and top-k
matrix = torch.rand(m, n)

# Get top-k indices
flat_indices = torch.topk(matrix.flatten(), k).indices
rows, cols = flat_indices // n, flat_indices % n

# Plot
#plt.figure(figsize=(6, 5))
plt.imshow(matrix, cmap='viridis')
plt.scatter(cols, rows, color='red', marker='o', label='Top-k values', s=1)
plt.colorbar()
plt.legend()
plt.title(f"Top-{k} values in the matrix")
plt.gca().invert_yaxis()  # Optional: make row 0 appear at top
plt.savefig('test.png')
