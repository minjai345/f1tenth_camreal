"""Process entry for both nodes: a stop signal only sets a flag, so destroy_node runs with a live context.
The console scripts start here, before the node modules import cv2, rclpy and onnxruntime."""
import ctypes
import os
import signal

# The flag is a list: append takes no lock, Event.set does and deadlocks on a signal nested inside it.
# SIGHUP: the terminal or SSH session closed. SIGQUIT: Ctrl+\.
STOP, SIGNALS = [], (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGQUIT)


def flag(signum, _frame):
    STOP.append(signum)


def catch():
    # Python's handlers raised KeyboardInterrupt mid-import (ros2 launch stopping one node as the other failed to start)
    # or inside destroy_node (the second SIGINT it forwards); rclpy's shut the context down before the final stop.
    for signum in SIGNALS:
        signal.signal(signum, flag)
    # PR_SET_PDEATHSIG: a parent that dies without signalling us (ros2 launch after SIGTERM, SIGKILL) sends SIGTERM.
    parent = os.getppid()
    ctypes.CDLL(None).prctl(1, ctypes.c_ulong(signal.SIGTERM))
    if os.getppid() != parent:
        STOP.append(signal.SIGTERM)


def waypoint_node():
    catch()
    from .waypoint_node import main
    main()


def pure_pursuit_node():
    catch()
    from .pure_pursuit_node import main
    main()


def run(node_class, make_executor, failure, args=None):
    import rclpy
    from rclpy.signals import SignalHandlerOptions
    catch()   # again for a direct main() call; the flag keeps what came while importing
    node = executor = None
    try:
        if STOP:
            return   # stopped while starting: nothing published yet, nothing to stop
        rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
        executor = make_executor()
        node = node_class()
        executor.add_node(node)
        while rclpy.ok() and not STOP:
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
        if executor is not None:
            executor.shutdown()
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        # Interpreter exit resets Python handlers to the default: a late Ctrl+C would end it by signal, logged by
        # ros2 launch as a crash ("process has died") after a clean stop.
        for signum in SIGNALS:
            signal.signal(signum, signal.SIG_IGN)
