import jax
import jax.numpy as jnp
from jax import grad
import time

print(f"JAX Version: {jax.__version__}")
print(f"Available Devices: {jax.devices()}")

# Define a simple function: f(x) = x^2 + 3x + 5
def f(x):
    return x**2 + 3*x + 5

# Analytical derivative: f'(x) = 2x + 3
# Let's use JAX to find the gradient automatically!
df_dx = grad(f)

# Test point
x_val = 4.0
print(f"\nFunction: f(x) = x^2 + 3x + 5")
print(f"Testing at x = {x_val}")

start_time = time.time()
# JAX calc
result = f(x_val)
gradient = df_dx(x_val)
end_time = time.time()

print(f"f({x_val}) = {result} (Expected: {x_val**2 + 3*x_val + 5})")
print(f"f'({x_val}) = {gradient} (Expected: {2*x_val + 3})")
print(f"Computation time (first run, includes compilation): {end_time - start_time:.6f}s")
