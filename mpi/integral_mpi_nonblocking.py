from mpi4py import MPI
from array import array
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

# Scatter: rank 0 sends one sub-interval range to each process.
# Data flow: rank 0 -> all ranks, each rank receives its own (start_i, end_i).
start_i, end_i = comm.scatter(ranges, root=0)

# Barrier: all ranks start timing together.
# Data flow: all ranks wait until every process reaches this point.
comm.Barrier()
start_time = time.time()

if rank == 0:
    recv_bufs = []
    recv_reqs = []

    for src in range(1, size):
        buf = array('d', [0.0])
        recv_bufs.append(buf)

        # Irecv: rank 0 posts non-blocking receives for workers' local results.
        # Data flow: rank src -> rank 0, receive can progress while rank 0 computes.
        recv_reqs.append(comm.Irecv(buf, source=src, tag=100))

    local_result = local_trapezoid(start_i, end_i, n)

    # Waitall: rank 0 waits until all non-blocking receives finish.
    # Data flow: synchronize completion of all worker -> rank 0 result transfers.
    MPI.Request.Waitall(recv_reqs)

    total_result = local_result + sum(buf[0] for buf in recv_bufs)

else:
    local_result = local_trapezoid(start_i, end_i, n)
    send_buf = array('d', [local_result])

    # Isend: worker sends local integral result to rank 0 without blocking immediately.
    # Data flow: current worker rank -> rank 0.
    req = comm.Isend(send_buf, dest=0, tag=100)

    # Wait: ensure the non-blocking send is complete before the buffer is released.
    # Data flow: current worker confirms its result has been transferred.
    req.Wait()

# Barrier: all ranks finish communication before timing ends.
# Data flow: all ranks wait for each other after non-blocking communication.
comm.Barrier()
elapsed = time.time() - start_time

if rank == 0:
    print(f"mpi processes = {size}")
    print("mode          = nonblocking Isend/Irecv")
    print(f"mpi result    = {total_result:.12f}")
    print(f"math.pi       = {math.pi:.12f}")
    print(f"error         = {abs(total_result - math.pi):.12e}")
    print(f"time          = {elapsed:.6f} s")
