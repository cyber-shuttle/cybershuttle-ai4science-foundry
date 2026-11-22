import numpy as np
import cebra

# Create fake neural data:
# 1000 time points, 20 neurons/features
X = np.random.randn(1000, 20)

# Create a small CEBRA model
model = cebra.CEBRA(
    model_architecture="offset10-model",
    batch_size=512,
    learning_rate=3e-4,
    max_iterations=100,
    output_dimension=3,
    verbose=True
)

# Train CEBRA on the fake data
model.fit(X)

# Generate embeddings
embedding = model.transform(X)

# Check the shapes
print("Input shape:", X.shape)
print("Embedding shape:", embedding.shape)