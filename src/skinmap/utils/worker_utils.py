"""Utilities for safely configuring DataLoader worker counts."""

import multiprocessing as mp
from functools import lru_cache

from loguru import logger


@lru_cache(maxsize=None)
def resolve_num_workers(requested: int) -> int:
    """Return a safe number of workers for DataLoaders.

    Some execution environments (CI, serverless, etc.) disallow creation of
    multiprocessing semaphores, which PyTorch relies on when spawning workers.
    This helper probes the environment once; if worker creation fails we fall
    back to single-threaded loading to avoid crashing tests or runs.
    """
    if requested <= 0:
        return 0

    try:
        ctx = mp.get_context()
        test_lock = ctx.Lock()
        test_lock.acquire()
        test_lock.release()
        return requested
    except (OSError, PermissionError) as exc:
        logger.warning(
            "Multiprocessing workers unavailable (%s); forcing num_workers=0", exc
        )
        return 0
