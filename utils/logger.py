import os
import time
import logging
import functools
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv

load_dotenv()  # So PIPEQL_AGENT_LOG_LEVEL from .env is read before the logger is set up

# All agents write to one file: <project root>/logs/pipeql_agent.log
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
LOG_FILE = os.path.join(LOG_DIR, "pipeql_agent.log")

# INFO logs every node and decision. Set PIPEQL_AGENT_LOG_LEVEL=DEBUG in .env to also log
# full prompts (e.g. the SQL prompt with the whole DB schema, which is very long).
LOG_LEVEL = os.environ.get("PIPEQL_AGENT_LOG_LEVEL", "INFO").upper()

# Example line:
# 2026-11-08 10:15:02 | INFO  | data_agent  | [router_node] Decision: route to 'sql'
LOG_FORMAT = "%(asctime)s | %(levelname)-5s | %(agent)-11s | %(message)s"


class _AgentNameFilter(logging.Filter):
    """Adds `agent` to each log line: the logger name without the 'pipeql_agent.' prefix."""
    def filter(self, record):
        record.agent = record.name.removeprefix("pipeql_agent.")
        return True


def _setup_base_logger() -> logging.Logger:
    """Configures the parent 'pipeql_agent' logger once. Every agent logger is a child of it."""
    base = logging.getLogger("pipeql_agent")

    if base.handlers:  # Already configured (this module can be imported many times)
        return base

    os.makedirs(LOG_DIR, exist_ok=True)

    # A new file is started when the log reaches 5 MB. The last 3 old files are kept
    # (pipeql_agent.log.1, .2, .3), so the logs can never fill up the disk.
    handler = RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"))
    handler.addFilter(_AgentNameFilter())

    base.addHandler(handler)
    base.setLevel(LOG_LEVEL)

    # Don't pass our logs up to Python's root logger. Otherwise they could also be
    # printed in the terminal and mess up the interactive PipeQL Agent screen.
    base.propagate = False

    return base


def get_logger(agent_name: str) -> logging.Logger:
    """
    Returns the logger for one agent, e.g. get_logger("sql_analyst").
    Its name appears in every log line, so you can see which agent wrote it.
    """
    _setup_base_logger()
    return logging.getLogger(f"pipeql_agent.{agent_name}")


def block(text, limit: int = 2000) -> str:
    """
    Formats a long or multi-line value (SQL, code, LLM answer) so it is easy to read
    in the log: it starts on a new line, is indented, and is cut off after `limit` characters.
    """
    text = str(text).strip()
    if len(text) > limit:
        text = text[:limit] + f"\n... [{len(text) - limit} more characters not shown]"
    return "\n" + "\n".join(f"        {line}" for line in text.splitlines())


def log_node(logger: logging.Logger):
    """
    Decorator for LangGraph nodes. It logs when the node starts, when it finishes
    and how long it took. If the node fails, it logs the full error before re-raising it.

    Usage:
        @log_node(logger)
        def my_node(state): ...
    """
    def decorator(func):
        @functools.wraps(func)  # Keeps the function name, which LangGraph uses as the node name
        def wrapper(*args, **kwargs):
            logger.info(f"[{func.__name__}] ▶ started")
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
            except Exception:
                elapsed = time.perf_counter() - start
                logger.exception(f"[{func.__name__}] ✗ failed after {elapsed:.2f}s")
                raise
            elapsed = time.perf_counter() - start
            logger.info(f"[{func.__name__}] ✓ finished in {elapsed:.2f}s")
            return result
        return wrapper
    return decorator
