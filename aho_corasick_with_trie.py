from collections import deque
from trie import TrieNode


class AhoCorasickWithTrie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True
        node.word = word

    def build_failure_links(self):
        # Build failure links for the Trie using BFS
        queue = deque()
        for char, child in self.root.children.items():
            child.fail = self.root
            queue.append(child)

        while queue:
            current = queue.popleft()
            for char, child in current.children.items():
                fail = current.fail
                while fail and char not in fail.children:
                    fail = fail.fail
                child.fail = fail.children[char] if fail else self.root
                queue.append(child)

    def search(self, grid):
        rows, cols = len(grid), len(grid[0])
        results = set()
        directions = [(0, 1), (1, 0), (1, 1), (-1, 1)]  # Right, Down, Diagonal, Reverse Diagonal

        # Iterate over each cell in the grid as a starting point
        for i in range(rows):
            for j in range(cols):
                # Skip empty cells (avoid unnecessary checks)
                if grid[i][j] not in self.root.children:
                    continue

                # Start DFS-like search for each direction
                node = self.root
                for dx, dy in directions:
                    x, y = i, j
                    while 0 <= x < rows and 0 <= y < cols:
                        char = grid[x][y]
                        
                        # Traverse the Trie using failure links
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
