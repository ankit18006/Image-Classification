import jax
import jax.numpy as jnp
import optax
import pickle
from flax.training import train_state
from tensorflow.keras.datasets import mnist
from model import AdvancedCNN

(train_images, train_labels), _ = mnist.load_data()

train_images = train_images.reshape(-1,28,28,1)/255.0

model = AdvancedCNN()
key = jax.random.PRNGKey(0)
params = model.init(key, jnp.ones((1,28,28,1)))['params']

tx = optax.adam(0.001)

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

for epoch in range(5):
    state = train_step(state, train_images, train_labels)
    print("Epoch:", epoch)

# Save model
with open("saved_model.pkl", "wb") as f:
    pickle.dump(state.params, f)