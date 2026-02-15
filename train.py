import jax
import jax.numpy as jnp
import optax
import pickle
import numpy as np
from flax.training import train_state
from tensorflow.keras.datasets import mnist
from model import AdvancedCNN

print("Loading MNIST dataset...")

(train_images, train_labels), _ = mnist.load_data()

train_images = train_images.astype(np.float32) / 255.0
train_images = train_images.reshape(-1, 28, 28, 1)

model = AdvancedCNN()
key = jax.random.PRNGKey(0)

dummy_input = jnp.ones((1,28,28,1))
params = model.init(key, dummy_input)['params']

tx = optax.adam(learning_rate=0.001)

state = train_state.TrainState.create(
    apply_fn=model.apply,
    params=params,
    tx=tx
)

def loss_fn(params, images, labels):
    logits = model.apply({'params': params}, images)
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits, labels
    ).mean()
    return loss

@jax.jit
def train_step(state, images, labels):
    grads = jax.grad(loss_fn)(state.params, images, labels)
    return state.apply_gradients(grads=grads)

print("Training started...")

# SMALL BATCH TRAINING (safe for PC)
batch_size = 128

for epoch in range(3):
    for i in range(0, len(train_images), batch_size):
        batch_images = train_images[i:i+batch_size]
        batch_labels = train_labels[i:i+batch_size]
        state = train_step(state, batch_images, batch_labels)

    print(f"Epoch {epoch+1} completed")

# SAVE MODEL
with open("saved_model.pkl", "wb") as f:
    pickle.dump(state.params, f)

print("✅ Model saved successfully!")