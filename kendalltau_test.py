from scipy import stats

# Testing the stats.kendalltau function

# Test with two lists where one element is swapped
tau, _ = stats.kendalltau([3,1,0,2,4], [3,1,0,4,2])
print("Kendall's tau for [3,1,0,2,4] and [3,1,0,4,2]:", tau)

# Test with two lists where the last two elements are swapped
tau, _ = stats.kendalltau([1,2,3,4,5], [1,2,3,5,4])
print("Kendall's tau for [1,2,3,4,5] and [1,2,3,5,4]:", tau)

# Test with two lists of floating point numbers
a = [0.5, 0.7, 0.2, 0.8, 0]
b = [0.6, 0.9, 0, 1, 0.4]
tau, _ = stats.kendalltau(a, b)
print("Kendall's tau for [0.5, 0.7, 0.2, 0.8, 0] and [0.6, 0.9, 0, 1, 0.4]:", tau)
