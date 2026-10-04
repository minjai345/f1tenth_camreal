"""Process entry for both nodes: Ctrl+C / SIGTERM only set a flag, so destroy_node runs with a live context."""
import signal
import threading
import rclpy
from rclpy.signals import SignalHandlerOptions


def run(node_class, make_executor, failure, args=None):
    # rclpy's own handlers shut the context down before destroy_node (the final stop could not be sent),
    # and the second SIGINT forwarded by ros2 launch raised KeyboardInterrupt inside destroy_node.
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())
    node, executor = None, make_executor()
    try:
        node = node_class()
        executor.add_node(node)
        while rclpy.ok() and not stop.is_set():
            executor.spin_once(timeout_sec=0.1)
    except Exception as exc:
        if node is None:
            print(f'{failure}: {exc}', flush=True)
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
