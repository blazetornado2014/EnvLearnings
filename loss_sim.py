import numpy as np

def calculate_cross_entropy():
    print("=== FEB 3: THE LOSS SCOREBOARD ===\n")

    # 1. The Model's Prediction (Softmax output from yesterday)
    # AI thinks: 80% iPhone, 15% Fruit, 5% Car
    predictions = np.array([0.80, 0.15, 0.05])
    
    # 2. The Truth (The Label from your dataset)
    # The dataset says it's 100% Fruit: [0, 1, 0]
    truth = np.array([0, 1, 0])

    # 3. Cross-Entropy Math: -sum(Truth * log(Prediction))
    # We only care about the probability the AI gave to the CORRECT answer.
    correct_prob = predictions[np.argmax(truth)]
    loss = -np.log(correct_prob)

    print(f"AI was {correct_prob:.1%} confident in the correct answer.")
    print(f"Calculated Loss: {loss:.4f}")
    
    if loss > 1.0:
        print("Status: HIGH LOSS. Backpropagation required to shift weights.")
    else:
        print("Status: LOW LOSS. Weights are stable.")

if __name__ == "__main__":
    calculate_cross_entropy()