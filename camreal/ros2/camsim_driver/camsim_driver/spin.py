"""Process entry for both nodes: a stop signal only sets a flag, so destroy_node runs with a live context."""
import ctypes
import os
import signal
import threading
import rclpy
from rclpy.signals import SignalHandlerOptions


def run(node_class, make_executor, failure, args=None):
    # rclpy's own handlers shut the context down before destroy_node (the final stop could not be sent),
    # and the second SIGINT forwarded by ros2 launch raised KeyboardInterrupt inside destroy_node.
    # SIGHUP: the terminal or SSH session closed. SIGQUIT: Ctrl+\.
    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGQUIT):
        signal.signal(signum, lambda *_: stop.set())
    # PR_SET_PDEATHSIG: a parent that dies without signalling us (ros2 launch after SIGTERM, SIGKILL) sends SIGTERM.
    parent = os.getppid()
    ctypes.CDLL(None).prctl(1, ctypes.c_ulong(signal.SIGTERM))
    if os.getppid() != parent:
        stop.set()
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node, executor = None, make_executor()
    try:
        node = node_class()
        executor.add_node(node)
        while rclpy.ok() and not stop.is_set():
            executor.spin_once(timeout_sec=0.1)
    except Exception as exc:
        if node is None:
            print(f'{failure}: {exc}', flush=True)
            if isinstance(exc, (ValueError, FileNotFoundError)):
                raise SystemExit(1) from None   # a setting to fix, named in the message: no traceback
        raise
    finally:
        pool = getattr(executor, '_executor', None)
        if pool is not None:   # Humble's MultiThreadedExecutor would run queued callbacks after shutdown
            pool.shutdown(wait=True)
        executor.shutdown()
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
