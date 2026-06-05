import time
import math

def f(x):
    return 4.0 / (1.0 + x * x)

def trapezoid(n):
    a = 0.0
    b = 1.0
    h = (b - a) / n
    total = 0.5 * (f(a) + f(b))

    for i in range(1, n):
        total += f(a + i * h)

    return total * h

if __name__ == "__main__":
    n = 10000000
    start = time.time()
    result = trapezoid(n)
    elapsed = time.time() - start

    print(f"serial result = {result:.12f}")
    print(f"math.pi       = {math.pi:.12f}")
    print(f"error         = {abs(result - math.pi):.12e}")
    print(f"time          = {elapsed:.6f} s")
