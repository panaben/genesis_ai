"""
Non-blocking CSV logger for Go1 real-robot data collection.

Writes rows from a background thread so the 50 Hz control loop is never blocked.
Each row captures all data needed for System Identification:
  timestamp, dof_pos[12], dof_vel[12], tau_est[12],
  imu_gyro[3], imu_quat[4], commands[3], actions[12]

Usage:
    logger = DataLogger("deploy_logs/run_001.csv")
    logger.log(timestamp, dof_pos, ...)   # non-blocking, called at 50 Hz
    logger.close()                         # flush and close on exit
"""

import csv
import queue
import threading
from pathlib import Path

import numpy as np

# CSV column header (56 columns total)
_HEADER: list[str] = (
    ["timestamp"]
    + [f"dof_pos_{i}" for i in range(12)]
    + [f"dof_vel_{i}" for i in range(12)]
    + [f"tau_est_{i}" for i in range(12)]
    + [f"imu_gyro_{i}" for i in range(3)]
    + [f"imu_quat_{i}" for i in range(4)]  # [w, x, y, z]
    + [f"cmd_{i}" for i in range(3)]        # [vx, vy, wz]
    + [f"action_{i}" for i in range(12)]
)


class DataLogger:
    """Thread-safe, non-blocking CSV logger.

    The control loop calls ``log()`` which enqueues the row without waiting.
    A background daemon thread consumes the queue and writes to disk.

    Parameters
    ----------
    path:
        Full path to the output CSV file.
    maxsize:
        Maximum number of rows buffered in memory before drops occur.
        At 50 Hz this gives ~20 seconds of buffer at the default of 1000.
    """

    def __init__(self, path: str, maxsize: int = 1000) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._queue: queue.Queue = queue.Queue(maxsize=maxsize)
        self._drop_count: int = 0
        self._thread = threading.Thread(target=self._writer, daemon=True, name="go1-logger")
        self._thread.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log(
        self,
        timestamp: float,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        tau_est: np.ndarray,
        imu_gyro: np.ndarray,
        imu_quat: np.ndarray,
        commands: np.ndarray,
        actions: np.ndarray,
    ) -> None:
        """Enqueue one row.  Returns immediately (non-blocking).

        If the queue is full the row is silently dropped to protect the
        control loop timing.
        """
        row = (
            [timestamp]
            + dof_pos.tolist()
            + dof_vel.tolist()
            + tau_est.tolist()
            + imu_gyro.tolist()
            + imu_quat.tolist()
            + commands.tolist()
            + actions.tolist()
        )
        try:
            self._queue.put_nowait(row)
        except queue.Full:
            self._drop_count += 1

    def close(self, timeout: float = 5.0) -> None:
        """Flush remaining rows and close the file.

        Blocks until the background thread finishes or *timeout* seconds pass.
        """
        self._queue.put(None)  # sentinel to stop the writer thread
        self._thread.join(timeout=timeout)
        if self._drop_count > 0:
            print(f"[logger] Warning: {self._drop_count} row(s) dropped due to queue full.")
        print(f"[logger] Saved to {self._path}  (thread alive={self._thread.is_alive()})")

    # ------------------------------------------------------------------
    # Background writer
    # ------------------------------------------------------------------

    def _writer(self) -> None:
        with self._path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(_HEADER)
            while True:
                row = self._queue.get()
                if row is None:
                    break
                writer.writerow(row)
                f.flush()
