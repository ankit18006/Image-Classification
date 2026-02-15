import jax.numpy as jnp
from flax import linen as nn

class AdvancedCNN(nn.Module):
    @nn.compact
    def __call__(self, x, training=False):
        x = nn.Conv(32, (3,3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (2,2), (2,2))

        x = nn.Conv(64, (3,3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (2,2), (2,2))

        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)

        return x