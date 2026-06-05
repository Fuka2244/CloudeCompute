from mpi4py import MPI
import time
import math

def f(x):
    return 4.0 / (1.0 + x * x)

def local_trapezoid(start_i, end_i, n):
    a = 0.0
    b = 1.0
    h = (b - a) / n
    local_sum = 0.0

    for i in range(start_i, end_i):
        x = a + i * h
        weight = 0.5 if i == 0 or i == n else 1.0
        local_sum += weight * f(x)

    return local_sum * h

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

n = 10000000

if rank == 0:
    chunk = n // size
    remainder = n % size
    ranges = []

    for r in range(size):
        if r < remainder:
            start_i = r * (chunk + 1)
            end_i = start_i + chunk + 1
        else:
            start_i = remainder * (chunk + 1) + (r - remainder) * chunk
            end_i = start_i + chunk
        ranges.append((start_i, end_i))
else:
    ranges = None

# Scatter: rank 0 sends one sub-interval index range to each process.
# Data flow: rank 0 -> rank 0..size-1, each rank receives (start_i, end_i).
start_i, end_i = comm.scatter(ranges, root=0)

# Barrier: all ranks synchronize before timing starts.
# Data flow: all ranks wait until every rank reaches this point.
comm.Barrier()
start_time = time.time()

local_result = local_trapezoid(start_i, end_i, n)

# Reduce: each rank sends its local integral result to rank 0.
# Data flow: rank 0..size-1 -> rank 0, rank 0 sums all local_result values.
total_result = comm.reduce(local_result, op=MPI.SUM, root=0)

# Barrier: all ranks synchronize after Reduce before measuring time.
# Data flow: all ranks wait until every rank finishes communication.
comm.Barrier()
elapsed = time.time() - start_time

if rank == 0:
    print(f"mpi processes = {size}")
    print(f"mpi result    = {total_result:.12f}")
    print(f"math.pi       = {math.pi:.12f}")
    print(f"error         = {abs(total_result - math.pi):.12e}")
    print(f"time          = {elapsed:.6f} s")
