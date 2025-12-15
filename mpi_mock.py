"""Simple MPI mock for single-process execution."""

import numpy as np

class MPIComm:
    """Mock MPI communicator for single-process execution."""
    
    def __init__(self):
        self._rank = 0
        self._size = 1
    
    def Get_rank(self):
        return self._rank
    
    def Get_size(self):
        return self._size
    
    def Allreduce(self, sendbuf, recvbuf, op=None):
        """For single process, just copy sendbuf to recvbuf."""
        if isinstance(sendbuf, np.ndarray) and isinstance(recvbuf, np.ndarray):
            recvbuf[:] = sendbuf[:]
        else:
            # Handle in-place modification
            if hasattr(recvbuf, '__setitem__'):
                recvbuf[:] = sendbuf[:]
            else:
                # If recvbuf is not writable, return sendbuf
                return sendbuf
        return recvbuf
    
    def allreduce(self, data, op=None):
        """For single process, just return the data (no reduction needed)."""
        # For single process, allreduce just returns the data as-is
        # since there's nothing to reduce across processes
        return data
    
    def bcast(self, data, root=0):
        """Broadcast data from root process to all processes.
        For single process, just return the data as-is."""
        # In single process mode, bcast just returns the data
        # since there's only one process (rank 0)
        return data

class MPIOp:
    """Mock MPI operation."""
    SUM = 'sum'

class MPI:
    """Mock MPI module."""
    COMM_WORLD = MPIComm()
    
    class Op:
        SUM = 'sum'
    
    # Make Op available at module level too
    SUM = 'sum'
