from trie import Trie


def backtracking_with_trie(grid, words):
    trie = Trie()
    for word in words:
        trie.insert(word)

    def dfs(x, y, node, path):
        if node.is_end_of_word:
            results.add(node.word)

        if not (0 <= x < rows and 0 <= y < cols) or (x, y) in path:
            return

        char = grid[x][y]
        if char not in node.children:
            return

        path.add((x, y))
        for dx, dy in directions:
            dfs(x + dx, y + dy, node.children[char], path)
        path.remove((x, y))

    rows, cols = len(grid), len(grid[0])
    results = set()
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    for i in range(rows):
        for j in range(cols):
            dfs(i, j, trie.root, set())
    return results
