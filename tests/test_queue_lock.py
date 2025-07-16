# tests/test_queue_lock.py
import os
import time
import pytest
from multiprocessing import Process
from queue_lock import QueueLock

def child_process_action(lock_file):
    lock = QueueLock(lock_file)
    lock.acquire()
    time.sleep(1) # Hold the lock for 1 second
    lock.release()
    exit(0)

@pytest.mark.filterwarnings("ignore:This process.*is multi-threaded, use of fork.*:DeprecationWarning")
def test_queue_lock_blocks_and_acquires():
    lock_file = "test_queue.lock"
    if os.path.exists(lock_file):
        os.remove(lock_file)

    child = Process(target=child_process_action, args=(lock_file,))
    child.start()
    
    time.sleep(0.2) # Give the child a moment to acquire the lock

    # This call will now block for ~0.8 seconds until the child releases the lock
    parent_lock = QueueLock(lock_file)
    parent_lock.acquire() 
    
    # If we get here, it means we successfully acquired the lock after waiting
    parent_lock.release()
    
    child.join(timeout=2)
    assert child.exitcode == 0
    
    if os.path.exists(lock_file):
        os.remove(lock_file)
