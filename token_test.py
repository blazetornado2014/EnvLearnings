import tiktoken

def run_lab_exercise():
    enc = tiktoken.get_encoding("cl100k_base")

    print("=== FEB 2: TOKENIZATION LAB (FIXED) ===\n")

    # 1. SUB-WORD LOGIC
    text = "Tokenization is simulation design."
    ids = enc.encode(text)
    parts = [enc.decode([i]) for i in ids]
    print(f"1. SUB-WORD LOGIC\nFragments: {parts}\n")

    # 2. THE SPACE BUG
    word1, word2 = "Apple", " Apple"
    id1, id2 = enc.encode(word1), enc.encode(word2)
    print(f"2. THE SPACE BUG\n'Apple': {id1} vs ' Apple': {id2}\n")

    # 3. THE VOCAB ATTIC (The "Ghost Tokens")
    print("3. EXPLORING THE VOCAB ATTIC")
    # We'll look at some very high IDs to see the 'weird' stuff
    test_ids = [100000, 150000, 160000, 199990]
    for i in test_ids:
        try:
            # We use errors='replace' to handle raw bytes that aren't valid UTF-8
            val = enc.decode([i], errors='replace')
            print(f"ID {i:7}: '{val}'")
        except Exception as e:
            print(f"ID {i}: [Error Decoding]")

    print("\nLab Complete. Press Enter to exit.")
    input() # This keeps the window open so you can read the results!

if __name__ == "__main__":
    run_lab_exercise()