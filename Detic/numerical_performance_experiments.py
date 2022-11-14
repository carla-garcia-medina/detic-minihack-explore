import os
import numpy as np

def numIslands(self, grid):
    def dfs(grid, i, j):
        if i < 0 or j < 0 or i >= len(grid) or j >= len(grid[0]) or grid[i][j] != '1':
            return 
        grid[i][j] = '#'
        dfs(grid, i+1, j)
        dfs(grid, i-1, j)
        dfs(grid, i, j+1)
        dfs(grid, i, j-1)
    count = 0
    if not grid:
        return 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == '1':
                dfs(grid, i, j)
                count += 1
                
    return count

def convert_hex_to_words():
    screen_descriptions_dir = 'minihack_datasets/MiniHack-River-Monster-v0/dataset_0/screen_descriptions'
    word_screen_descriptions_dir = 'minihack_datasets/MiniHack-River-Monster-v0/dataset_0/word_screen_descriptions'
        
    for file in os.scandir(screen_descriptions_dir):
        matrix = np.load(file)
        word_matrix = np.chararray(matrix.shape)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                pixel = matrix[i][j]
                message = ''.join([chr(hex) for hex in pixel if hex>0])
                word_matrix[i, j] = message

        np.save(word_screen_descriptions_dir+file.name, word_matrix)

convert_hex_to_words()