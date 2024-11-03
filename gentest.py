import random

def create_sokoban_maze(maze_size, num_stones):
    maze = [['#'] * (maze_size + 2) for _ in range(maze_size + 2)]
    
    # Create walls and open spaces
    for i in range(1, maze_size + 1):
        for j in range(1, maze_size + 1):
            if random.random() < 0.2:  # 20% chance of a wall
                maze[i][j] = '#'
            else:
                maze[i][j] = ' '
    
    # Position Ares
    ares_x, ares_y = random.randint(1, maze_size), random.randint(1, maze_size)
    maze[ares_x][ares_y] = '@'
    
    # Place stones
    stone_positions = []
    for _ in range(num_stones):
        while True:
            x, y = random.randint(1, maze_size), random.randint(1, maze_size)
            if maze[x][y] == ' ':
                maze[x][y] = '$'
                stone_positions.append((x, y))
                break

    # Create switch positions
    switch_positions = []
    for x, y in stone_positions:
        # Place a switch next to each stone
        if maze[x][y + 1] == ' ':
            maze[x][y + 1] = '.'
            switch_positions.append((x, y + 1))
        elif maze[x][y - 1] == ' ':
            maze[x][y - 1] = '.'
            switch_positions.append((x, y - 1))
        elif maze[x + 1][y] == ' ':
            maze[x + 1][y] = '.'
            switch_positions.append((x + 1, y))
        elif maze[x - 1][y] == ' ':
            maze[x - 1][y] = '.'
            switch_positions.append((x - 1, y))
    
    return maze, stone_positions, switch_positions

def generate_input_file(index):
    maze_size = random.randint(5, 7)  # Size of the maze
    num_stones = random.randint(1, 4)  # Number of stones
    weights = [random.randint(1, 5) for _ in range(num_stones)]  # Weights for stones
    
    maze, stones, switches = create_sokoban_maze(maze_size, num_stones)
    
    # Prepare file content
    weights_line = ' '.join(map(str, weights))
    maze_lines = [''.join(row) for row in maze]
    
    # Save to file
    filename = f'input-{index:02d}.txt'
    with open(filename, 'w') as f:
        f.write(weights_line + '\n')
        f.write('\n'.join(maze_lines) + '\n')

# Generate 10 input files
for i in range(1, 11):
    generate_input_file(i)

print("Generated 10 Sokoban input files.")
