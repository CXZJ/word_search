def backtracking_without_trie(grid, words):
    def dfs(x, y, path, current_word):
        if current_word in words_set:
            results.add(current_word)

        if not (0 <= x < rows and 0 <= y < cols) or (x, y) in path:
            return

        char = grid[x][y]
        current_word += char
        if not any(word.startswith(current_word) for word in words):
            return

        path.add((x, y))
        for dx, dy in directions:
            dfs(x + dx, y + dy, path, current_word)
        path.remove((x, y))

    rows, cols = len(grid), len(grid[0])
    results = set()
    words_set = set(words)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    for i in range(rows):
        for j in range(cols):
            dfs(i, j, set(), "")
    return results
