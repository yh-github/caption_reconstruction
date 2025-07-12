# src/queue_lock.py
import os
import fcntl
import logging
import time

class QueueLock:
    """
    A simple filesystem-based lock using a blocking flock call.
    """
    def __init__(self, lock_file_path="queue.lock"):
        self.lock_file_path = lock_file_path
        self._lock_file = None

    def acquire(self):
        """
        Acquires an exclusive lock on the file. This call will block
        indefinitely until the lock is available.
        """
        logging.info(f"Process {os.getpid()} attempting to acquire lock...")
        self._lock_file = open(self.lock_file_path, 'w')
        
        # This is a blocking call. The script will pause here until the lock is acquired.
        fcntl.flock(self._lock_file, fcntl.LOCK_EX)
        
        self._lock_file.write(str(os.getpid()))
        self._lock_file.flush()
        logging.info(f"Lock acquired by PID {os.getpid()}.")

    def release(self):
        """Releases the lock."""
        if self._lock_file and not self._lock_file.closed:
            fcntl.flock(self._lock_file, fcntl.LOCK_UN)
            self._lock_file.close()
            self._lock_file = None
            logging.info(f"Lock released by PID {os.getpid()}.")
