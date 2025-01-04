from collections import deque
import time
import random
import string

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
        self.word = None
        self.fail = None

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word.lower():
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True
        node.word = word.lower()

class AhoCorasickWithTrie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word.lower():
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True
        node.word = word.lower()

    def build_failure_links(self):
        queue = deque()
        
        # Set failure for depth 1 nodes to root
        for child in self.root.children.values():
            child.fail = self.root
            queue.append(child)
            
        while queue:
            current = queue.popleft()
            
            # Process all children of current node
            for char, child in current.children.items():
                queue.append(child)
                
                # Find failure link
                failure = current.fail
                while failure and char not in failure.children:
                    failure = failure.fail
                child.fail = failure.children[char] if failure else self.root

    def search(self, grid):
        rows, cols = len(grid), len(grid[0])
        results = set()
        directions = [(0, 1), (1, 0), (1, 1), (-1, 1),
                     (0, -1), (-1, 0), (-1, -1), (1, -1)]

        for i in range(rows):
            for j in range(cols):
                for dx, dy in directions:
                    x, y = i, j
                    node = self.root
                    
                    while 0 <= x < rows and 0 <= y < cols:
                        char = grid[x][y].lower()
                        
                        # Follow failure links
                        while node and char not in node.children:
                            node = node.fail
                            
                        if not node:
                            node = self.root
                            break
                            
                        node = node.children[char]
                        if node.is_end_of_word:
                            results.add(node.word)
                            
                        x, y = x + dx, y + dy
                        
        return results

class AhoCorasickWithoutTrie:
    def __init__(self):
        self.words = []
        self.words_set = set()
        self.max_word_length = 0

    def insert(self, word):
        word = word.lower()
        self.words.append(word)
        self.max_word_length = max(self.max_word_length, len(word))

    def build_failure_links(self):
        self.words_set = set(self.words)

    def search(self, grid):
        rows, cols = len(grid), len(grid[0])
        results = set()
        directions = [(0, 1), (1, 0), (1, 1), (-1, 1),
                     (0, -1), (-1, 0), (-1, -1), (1, -1)]

        for i in range(rows):
            for j in range(cols):
                for dx, dy in directions:
                    x, y = i, j
                    word = ""
                    while 0 <= x < rows and 0 <= y < cols and len(word) <= self.max_word_length:
                        word += grid[x][y].lower()
                        if word in self.words_set:
                            results.add(word)
                        x, y = x + dx, y + dy
        return results

def backtracking_with_trie(grid, words):
    trie = Trie()
    for word in words:
        trie.insert(word)

    def dfs(x, y, node, path, dx, dy):
        if not (0 <= x < rows and 0 <= y < cols):
            return
            
        char = grid[x][y].lower()
        if char not in node.children:
            return
            
        current_node = node.children[char]
        if current_node.is_end_of_word:
            results.add(current_node.word)

        if (x, y) not in path:
            path.add((x, y))
            next_x, next_y = x + dx, y + dy
            dfs(next_x, next_y, current_node, path, dx, dy)
            path.remove((x, y))

    rows, cols = len(grid), len(grid[0])
    results = set()
    directions = [(0, 1), (1, 0), (1, 1), (-1, 1),
                 (0, -1), (-1, 0), (-1, -1), (1, -1)]

    for i in range(rows):
        for j in range(cols):
            for dx, dy in directions:
                dfs(i, j, trie.root, set(), dx, dy)
    return results

def backtracking_without_trie(grid, words):
    def dfs(x, y, path, current_word, dx, dy):
        if not (0 <= x < rows and 0 <= y < cols):
            return
            
        if len(current_word) > max_word_length:
            return
            
        new_word = current_word + grid[x][y].lower()
        if new_word in words_set:
            results.add(new_word)
            
        if not any(word.startswith(new_word) for word in words_set):
            return

        if (x, y) not in path:
            path.add((x, y))
            next_x, next_y = x + dx, y + dy
            dfs(next_x, next_y, path, new_word, dx, dy)
            path.remove((x, y))

    rows, cols = len(grid), len(grid[0])
    results = set()
    words_set = set(word.lower() for word in words)
    max_word_length = max(len(word) for word in words)
    directions = [(0, 1), (1, 0), (1, 1), (-1, 1),
                 (0, -1), (-1, 0), (-1, -1), (1, -1)]

    for i in range(rows):
        for j in range(cols):
            for dx, dy in directions:
                dfs(i, j, set(), "", dx, dy)
    return results

def generate_word_search(words, grid_size=10):
    # Sort words by length (longest first) and filter out words too long for grid
    words = [w.lower() for w in words if len(w) <= grid_size]
    words = sorted(words, key=len, reverse=True)
    
    grid = [[' ' for _ in range(grid_size)] for _ in range(grid_size)]
    placed_words = []
    
    directions = [(0, 1), (1, 0), (1, 1), (-1, 1),
                 (0, -1), (-1, 0), (-1, -1), (1, -1)]

    def can_place_word(word, row, col, dx, dy):
        word_len = len(word)
        for i in range(word_len):
            new_row, new_col = row + dx * i, col + dy * i
            if not (0 <= new_row < grid_size and 0 <= new_col < grid_size):
                return False
            if grid[new_row][new_col] != ' ' and grid[new_row][new_col] != word[i]:
                return False
        return True

    def place_word(word, row, col, dx, dy):
        for i, char in enumerate(word):
            new_row, new_col = row + dx * i, col + dy * i
            grid[new_row][new_col] = char

    # Try to place each word with smart positioning
    for word in words:
        placed = False
        max_attempts = grid_size * grid_size * len(directions)
        attempts = 0

        while not placed and attempts < max_attempts:
            # Try strategic positions first
            if attempts < max_attempts // 3:
                positions = [
                    (0, 0), (0, grid_size-1),  # corners
                    (grid_size-1, 0), (grid_size-1, grid_size-1),
                    (grid_size//2, grid_size//2),  # center
                    (grid_size//4, grid_size//4),  # quarter points
                    (grid_size//4, 3*grid_size//4),
                    (3*grid_size//4, grid_size//4),
                    (3*grid_size//4, 3*grid_size//4)
                ]
                row, col = random.choice(positions)
            else:
                row = random.randint(0, grid_size - 1)
                col = random.randint(0, grid_size - 1)

            direction = random.choice(directions)
            dx, dy = direction

            if can_place_word(word, row, col, dx, dy):
                place_word(word, row, col, dx, dy)
                placed = True
                placed_words.append(word)
            attempts += 1

    # Fill empty spaces with random letters
    for i in range(grid_size):
        for j in range(grid_size):
            if grid[i][j] == ' ':
                grid[i][j] = random.choice(string.ascii_lowercase)

    return grid, placed_words

def evaluate(grid, words):
    algorithms = [
        ("Aho-Corasick (Trie)", AhoCorasickWithTrie()),
        ("Aho-Corasick", AhoCorasickWithoutTrie()),
        ("Backtracking (Trie)", "backtracking_with_trie"),
        ("Backtracking", "backtracking_without_trie")
    ]

    print("\nGrid:")
    print("-" * (len(grid[0]) * 2 + 1))
    for row in grid:
        print("|" + " ".join(row) + "|")
    print("-" * (len(grid[0]) * 2 + 1))

    print("\nPerformance Comparison:")
    print("-" * 85)
    print(f"{'Algorithm':<20} {'Time (s)':<12} {'Words Found':<12} {'Sample Words Found':<40}")
    print("-" * 85)
    
    for name, algo in algorithms:
        if isinstance(algo, str):
            start_time = time.time()
            if algo == "backtracking_with_trie":
                found_words = backtracking_with_trie(grid, words)
            else:
                found_words = backtracking_without_trie(grid, words)
            end_time = time.time()
        else:
            for word in words:
                algo.insert(word)
            if hasattr(algo, 'build_failure_links'):
                algo.build_failure_links()
            start_time = time.time()
            found_words = algo.search(grid)
            end_time = time.time()
            
        exec_time = end_time - start_time
        sample = list(found_words)[:4] if found_words else []
        print(f"{name:<20} {exec_time:<12.6f} {len(found_words):<12} {', '.join(sample):<40}")
    
    print("-" * 85)

def main():
    # Test case 1: Small number of short words
    simple_words = ["cat", "rat", "car", "art", "arc", "cart"]
    
    # Test case 2: More words with overlapping patterns
    overlapping_words = [
        "cat", "catch", "cater", "caters",
        "rat", "rate", "rates", "rating",
        "car", "cart", "carts", "carting",
        "art", "arts", "artist", "artistic"
    ]
    
    # Test case 3: Dictionary words (filtered by length)
    try:
        with open('/usr/share/dict/words', 'r') as f:
            all_words = f.readlines()
        dictionary_words = [word.strip().lower() for word in all_words 
                          if 3 <= len(word.strip()) <= 8][:50]
    except:
        dictionary_words = [
            "python", "program", "code", "algorithm",
            "search", "find", "word", "grid", "test",
            "computer", "data", "string", "array"
        ]

    test_cases = {
        "Simple Test (6 words)": simple_words,
        "Overlapping Patterns (16 words)": overlapping_words,
        "Dictionary Words": dictionary_words
    }

    sizes = [10, 15, 20]

    for test_name, words in test_cases.items():
        print(f"\n=== {test_name} ===")
        
        for size in sizes:
            print(f"\n--- Grid Size: {size}x{size} ---")
            
            grid, placed_words = generate_word_search(words, grid_size=size)
            
            print("\nGrid Preview (first 5 rows):")
            for row in grid[:5]:
                print(' '.join(row))
            
            # Use only placed words for evaluation
            evaluate(grid, placed_words)

if __name__ == "__main__":
    main()
