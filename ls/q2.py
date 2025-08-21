import numpy as np

arr = np.random.uniform(0, 10, 20)
rounded_arr = np.round(arr, 2)
print("Original Array : \n", rounded_arr)

mini = np.min(rounded_arr)
maxi = np.max(rounded_arr)
median = np.median(rounded_arr)
print(f"Minimum: {mini}, Maximum: {maxi}, Median: {median}")

modified_arr = np.where(rounded_arr < 5, np.round(rounded_arr**2, 2), rounded_arr)
print("Modified Array :\n", modified_arr)

def numpy_alternate_sort(array):
    sorted_arr = np.sort(array)
    result = []
    left = 0
    right = len(sorted_arr) - 1
    while left <= right:
        if left == right:
            result.append(sorted_arr[left])
        else:
            result.extend([sorted_arr[left], sorted_arr[right]])
        left += 1
        right -= 1
    return np.array(result)
new_array = numpy_alternate_sort(rounded_arr)
