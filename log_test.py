import numpy as np

# Let's see how the 'Penalty' grows as confidence drops
confidences = [0.99, 0.9, 0.7, 0.5, 0.3, 0.1, 0.01]

print("Confidence | Loss (Penalty)")
print("-----------|---------------")
for c in confidences:
    loss = -np.log(c)
    print(f"{c:10.0%} | {loss:.4f}")