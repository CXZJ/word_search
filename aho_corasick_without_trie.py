from collections import deque


class AhoCorasickWithoutTrie:
    def __init__(self):
        self.words = []

    def insert(self, word):
        self.words.append(word)

    def build_failure_links(self):
        self.words_set = set(self.words)

    def search(self, grid):
        rows, cols = len(grid), len(grid[0])
        results = set()
        directions = [(0, 1), (1, 0), (1, 1), (-1, 1)]  # Right, Down, Diagonal, Reverse Diagonal

        for i in range(rows):
            for j in range(cols):
                for dx, dy in directions:
                    x, y = i, j
                    word = ""
                    while 0 <= x < rows and 0 <= y < cols:
                        word += grid[x][y]
                        if word in self.words_set:
                            results.add(word)
                        x, y = x + dx, y + dy
        return results
