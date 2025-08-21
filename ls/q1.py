import numpy as np

np.random.seed(0)  
arr = np.random.randint(1, 51, size=(5, 4))
print("Array:\n", arr)

element_cross = [arr[i, -i-1] for i in range(min(arr.shape))]

row_max = np.max(arr, axis=1)
print("Max value in each row:", row_max)

mean_value = np.mean(arr)
sorted = arr[arr <= mean_value]

def numpy_boundary_traversal(matrix):
    if matrix.size == 0:
        return []

    top = matrix[0, :]
    right = matrix[1:-1, -1]
    bottom = matrix[-1, ::-1]
    left = matrix[-2:0:-1, 0]

    boundary = np.concatenate((top, right, bottom, left)).tolist()
    return boundary

peripheral_elements = numpy_boundary_traversal(arr)
