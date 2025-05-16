# import logging, os, sys, builtins
# from logging.handlers import RotatingFileHandler

# # def setup_logger(name: str, log_dir: str, rank: int | None = None, level=logging.INFO, max_mb: int = 10, backups: int = 3) -> logging.Logger:
# def setup_logger(name: str, log_dir: str, rank: int|None=None, *, full_detail: bool=False, max_mb: int=10, backups: int=3) -> logging.Logger:
#     """
#     • Each process calls this **once**.  
#     • Creates `<log_dir>/<name>[_<rank>].log` and a matching stdout stream.
#     """
#     os.makedirs(log_dir, exist_ok=True)
#     suffix = f"_{rank}" if rank is not None else ""
#     path   = os.path.join(log_dir, f"{name}{suffix}.log")

#     logger = logging.getLogger(f"{name}{suffix}")
#     if logger.handlers:          # already initialised in this process
#         return logger

#     logger.setLevel(logging.DEBUG if full_detail else logging.INFO)
#     fmt = logging.Formatter("%(asctime)s | %(processName)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

#     fh = RotatingFileHandler(path, maxBytes=max_mb * 1_048_576, backupCount=backups)
#     fh.setFormatter(fmt)
#     sh = logging.StreamHandler(sys.stdout)
#     sh.setFormatter(fmt)

#     logger.addHandler(fh)
#     logger.addHandler(sh)
#     logger.propagate = False     # don’t double-print via root logger
#     return logger


# def hijack_print(logger: logging.Logger):
#     """Route every bare `print()` to logger.info – handy for legacy code."""
#     builtins.print = lambda *a, **k: logger.info(" ".join(map(str, a)))
