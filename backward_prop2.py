import numpy as np

def deep_chain_simulation():
    print("=== THE 5-LAYER BLAME CHAIN ===\n")

    # 1. SETUP: 5 Layers of weights
    x = 1.0
    weights = [0.6, 0.7, 0.8, 0.9, 0.5] # w1, w2, w3, w4, w5
    target = 1.0
    lr = 0.1

    # 2. FORWARD PASS: Calculating the prediction
    # We pass the signal through all 5 layers
    activations = [x]
    for w in weights:
        next_val = activations[-1] * w
        activations.append(next_val)
    
    prediction = activations[-1]
    loss = 0.5 * (target - prediction)**2
    print(f"Initial Prediction: {prediction:.4f} | Target: {target}")
    print(f"Initial Loss: {loss:.4f}\n")

    # 3. BACKWARD PASS: The Chain Rule (The Blame Game)
    # The 'Error Report' from the Principal
    error_signal = -(target - prediction) 

    # To find the blame for weight[i], we multiply:
    # (Error Signal) * (Input to that layer) * (All weights that came AFTER it)
    
    gradients = [0] * 5
    
    # Blame for Weight 5 (The last one)
    gradients[4] = error_signal * activations[4]
    
    # Blame for Weight 4
    gradients[3] = error_signal * weights[4] * activations[3]
    
    # Blame for Weight 1 (The first one - The longest chain!)
    # grad_w1 = error_signal * w5 * w4 * w3 * w2 * input
    gradients[0] = error_signal * weights[4] * weights[3] * weights[2] * weights[1] * activations[0]

    print("Calculated Blame (Gradients):")
    for i, g in enumerate(gradients):
        print(f"  Layer {i+1}: {g:.4f}")

    # 4. UPDATE: Turn the knobs
    for i in range(5):
        weights[i] -= lr * gradients[i]

    # 5. VERIFY
    final_val = x
    for w in weights:
        final_val *= w
    print(f"\nNew Prediction: {final_val:.4f} | New Loss: {0.5 * (target - final_val)**2:.4f}")

if __name__ == "__main__":
    deep_chain_simulation()