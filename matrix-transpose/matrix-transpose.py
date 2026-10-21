import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    """
    Returns the transposed matrix as a NumPy array.
    """
    i = np.shape(A)[0]
    j = np.shape(A)[1]

    transpose = np.zeros((j,i), dtype=float)

    for col in range(j):
        for row in range(i):
            transpose[col][row] = A[row][col] 
            
    return transpose
