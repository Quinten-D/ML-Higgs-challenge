from implementations import *


GAMMA = 0.1
MAX_ITERS = 2

y = np.array([[0.],
              [1.],
              [1.]])
tx = np.array([[2.3, 3.2],
               [1., 0.1],
               [1.4, 2.3]])
initial_w = np.array([[0.5], [1.]])

lambda_ = 1.0
expected_w = np.array([[0.409111], [0.843996]])
expected_w = np.array([[0.463156], [0.939874]])
y = (y > 0.2) * 1.0
print(y)
w, loss = logistic_regression(y, tx, expected_w, 0, GAMMA)
print("\nresults:")
print(loss)
print(w)