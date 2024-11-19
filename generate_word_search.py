import random
import string

# Directions for placing words: (row_offset, col_offset)
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]  # Up, Down, Left, Right, Diagonal directions

def generate_empty_grid(size=10):
    """Generates a 10x10 grid filled with empty spaces."""
    return [[' ' for _ in range(size)] for _ in range(size)]

def generate_random_letter():
    """Generates a random uppercase letter."""
    return random.choice(string.ascii_uppercase)

def can_place_word(grid, word, row, col, direction):
    """Checks if a word can be placed at the specified location in the grid."""
    rows, cols = len(grid), len(grid[0])
    word_len = len(word)
    dx, dy = direction

    # Check if the word fits in the grid in the given direction
    for i in range(word_len):
        new_row, new_col = row + dx * i, col + dy * i
        if not (0 <= new_row < rows and 0 <= new_col < cols) or grid[new_row][new_col] != ' ':
            return False
    return True

def place_word(grid, word, row, col, direction):
    """Places a word in the grid at the specified location and direction."""
    dx, dy = direction
    for i in range(len(word)):
        new_row, new_col = row + dx * i, col + dy * i
        grid[new_row][new_col] = word[i]

def generate_word_search(words, grid_size=10):
    """Generates a random word search puzzle with the given words."""
    grid = generate_empty_grid(grid_size)

    # Place each word in the grid
    for word in words:
        placed = False
        while not placed:
            row = random.randint(0, grid_size - 1)
            col = random.randint(0, grid_size - 1)
            direction = random.choice(DIRECTIONS)

            if can_place_word(grid, word, row, col, direction):
                place_word(grid, word, row, col, direction)
                placed = True

    # Fill in the remaining empty spaces with random letters
    for i in range(grid_size):
        for j in range(grid_size):
            if grid[i][j] == ' ':
                grid[i][j] = generate_random_letter()

    return grid

def print_grid(grid):
    """Prints the grid in a readable format."""
    for row in grid:
        print(' '.join(row))

if __name__ == "__main__":
    words = ["CAT", "RAT", "ART", "TAT", "CAR", "ACT"]
    grid = generate_word_search(words, grid_size=10)
    print_grid(grid)
