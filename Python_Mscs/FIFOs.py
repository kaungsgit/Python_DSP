from collections import deque
import numpy as np
from IPython import get_ipython
ipython = get_ipython()

# FIFO with a deque
# Using a deque is most effecicent when you only need access to
# the first and last values in a container (such as a FIFO)
print("Deque FIFO")
x = deque(range(5), 5)
print(x)
x.append(5)
print(x)

print("\n\n")

# FIFO with a List
# need to append and pop. Appending to a list is efficient
# as is pop() from the end (as done for a stack) but pop(0)
# from the start of a list is not because all elements must
# be copied and moved
print("List FIFO")
x = list(range(5))
print(x)
x.append(5)
print(x)
x.pop(0)
print(x)

print("\n\n")

# FIFO with NumPy
print("NumPy FIFO")
x = np.array(range(5))
print(x)
# shift
x[:-1] = x[1:]
print(x)
# add new value
x[-1] = 5
print(x)
