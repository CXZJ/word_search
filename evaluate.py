import time
from tqdm import tqdm
from aho_corasick_with_trie import AhoCorasickWithTrie
from aho_corasick_without_trie import AhoCorasickWithoutTrie
from backtracking_with_trie import backtracking_with_trie
from backtracking_without_trie import backtracking_without_trie


def evaluate(grid, words):
    # Aho-Corasick with Trie
    ac_with_trie = AhoCorasickWithTrie()
    for word in words:
        ac_with_trie.insert(word)
    ac_with_trie.build_failure_links()

    print("Running Aho-Corasick with Trie...")
    start_time = time.time()
    ac_with_results = ac_with_trie.search(grid)
    ac_with_time = time.time() - start_time
    print(f"Aho-Corasick with Trie completed in {ac_with_time:.6f} seconds.")

    # Aho-Corasick without Trie
    ac_without_trie = AhoCorasickWithoutTrie()
    for word in words:
        ac_without_trie.insert(word)
    ac_without_trie.build_failure_links()

    print("Running Aho-Corasick without Trie...")
    start_time = time.time()
    ac_without_results = ac_without_trie.search(grid)
    ac_without_time = time.time() - start_time
    print(f"Aho-Corasick without Trie completed in {ac_without_time:.6f} seconds.")

    # Backtracking with Trie
    print("Running Backtracking with Trie...")
    start_time = time.time()
    bt_with_results = backtracking_with_trie(grid, words)
    bt_with_time = time.time() - start_time
    print(f"Backtracking with Trie completed in {bt_with_time:.6f} seconds.")

    # Backtracking without Trie
    print("Running Backtracking without Trie...")
    start_time = time.time()
    bt_without_results = backtracking_without_trie(grid, words)
    bt_without_time = time.time() - start_time
    print(f"Backtracking without Trie completed in {bt_without_time:.6f} seconds.")

    return {
        "Aho-Corasick with Trie": {"results": ac_with_results, "time": ac_with_time},
        "Aho-Corasick without Trie": {"results": ac_without_results, "time": ac_without_time},
        "Backtracking with Trie": {"results": bt_with_results, "time": bt_with_time},
        "Backtracking without Trie": {"results": bt_without_results, "time": bt_without_time},
    }


if __name__ == "__main__":
    grid = [
        ['c', 'a', 't'],
        ['r', 'a', 't'],
        ['t', 'a', 'c']
    ]
    words = ["cat", "rat", "car", "art", "tat"]

    print("Starting evaluation...")
    results = evaluate(grid, words)
    
    for method, data in results.items():
        print(f"{method}:\n  Words Found: {data['results']}\n  Time Taken: {data['time']:.6f} seconds\n")
