import logging, pathlib
from rich.logging import RichHandler


def setup_logger(name: str, log_dir: str, level: int = logging.INFO, to_console: bool = False) -> logging.Logger:
    path = pathlib.Path(log_dir) / f"{name}.log"; path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name); logger.setLevel(level)
    logger.propagate = False # keep noise out of the root logger
    fh = logging.handlers.RotatingFileHandler(path, maxBytes=10_000_000, backupCount=5, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(fh)
    if to_console: # optional colourful console for the *driver* only
        ch = RichHandler(rich_tracebacks=True, markup=True)
        ch.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(ch)
    return logger
