import numpy as np

def run_backprop_foundation():
    print("=== THE CHAIN RULE ENGINE ===\n")

    # 1. SETUP: Input -> Weight1 -> Weight2 -> Output
    x = 1.0          # Input (The Prompt)
    w1 = 0.5         # Layer 1 Weight
    w2 = 0.7         # Layer 2 Weight
    target = 1.0     # The 'Correct' answer we want (The Truth)

    # 2. FORWARD PASS: Calculating the prediction
    layer1 = x * w1         # 0.5
    prediction = layer1 * w2 # 0.35
    
    loss = 0.5 * (target - prediction)**2 # 0.5 * (0.65)^2 = 0.211
    print(f"Prediction: {prediction:.3f} | Target: {target} | Initial Loss: {loss:.4f}")

    # 3. BACKWARD PASS: The 'Chain' of blame
    # dLoss/dPrediction (How much does loss change if prediction changes?)
    error_signal = -(target - prediction) # -0.65
    
    # dLoss/dw2 = error_signal * layer1
    grad_w2 = error_signal * layer1 # -0.325
    
    # dLoss/dw1 = error_signal * w2 * x (This is the 'Chain' part!)
    grad_w1 = error_signal * w2 * x # -0.455

    # 4. UPDATE: Gradient Descent
    lr = 0.1
    new_w1 = w1 - (lr * grad_w1)
    new_w2 = w2 - (lr * grad_w2)

    # 5. VERIFY: Did it learn?
    new_pred = (x * new_w1) * new_w2
    new_loss = 0.5 * (target - new_pred)**2
    print(f"New Prediction: {new_pred:.3f} | New Loss: {new_loss:.4f}")
    print("\nInsight: Both weights changed because the error flowed backward.")

if __name__ == "__main__":
    run_backprop_foundation()