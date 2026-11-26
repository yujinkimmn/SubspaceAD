"""Compact terminal progress with a complete, plain-text run.log."""
import logging
import sys
from contextlib import contextmanager

try:
    from tqdm import tqdm
except ImportError:
    # Keep existing offline installations usable until requirements are updated.
    tqdm = None

LOG = logging.getLogger("csam_void")
_stream = None
_enabled = False
_configured = False


class ConsoleFilter(logging.Filter):
    def __init__(self, verbose=False):
        super().__init__()
        self.verbose = verbose

    def filter(self, record):
        return self.verbose or record.levelno >= logging.WARNING or getattr(record, "console", False)


class ProgressConsole(logging.StreamHandler):
    """Write messages without corrupting an active tqdm line."""
    def __init__(self, stream, enabled):
        super().__init__(stream)
        self.enabled = enabled

    def emit(self, record):
        if not self.enabled:
            return super().emit(record)
        try:
            tqdm.write(self.format(record), file=self.stream)
            self.flush()
        except Exception:
            self.handleError(record)


def setup_logging(root, *, progress_mode="auto", verbose=False, stream=None):
    global _stream, _enabled, _configured
    if progress_mode not in ("auto", "on", "off"):
        raise ValueError("progress_mode must be auto/on/off")
    _stream = sys.stdout if stream is None else stream
    requested = progress_mode == "on" or (progress_mode == "auto" and bool(getattr(_stream, "isatty", lambda: False)()))
    _enabled = requested and tqdm is not None
    _configured = True
    LOG.setLevel(logging.INFO)
    LOG.propagate = False
    for handler in list(LOG.handlers):
        LOG.removeHandler(handler)
        handler.close()
    file_handler = logging.FileHandler(root / "run.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
    console = ProgressConsole(_stream, _enabled)
    console.setFormatter(logging.Formatter("%(message)s"))
    console.addFilter(ConsoleFilter(verbose))
    LOG.addHandler(file_handler)
    LOG.addHandler(console)
    if requested and tqdm is None:
        LOG.warning("tqdm is not installed; using compact text progress. Install models/csam_void/requirements.txt to enable bars.")


def console_info(message, *args):
    """INFO lines intended for both human-readable console and full log."""
    LOG.info(message, *args, extra={"console": True})


def epoch_note(message, *args):
    """With bars disabled, keep only one short line per completed epoch."""
    if _configured and not _enabled:
        console_info(message, *args)


class QuietProgress:
    """Same small interface used by the loops when bars are disabled/unavailable."""
    def update(self, n=1):
        pass

    def set_description_str(self, desc, refresh=True):
        pass

    def set_postfix(self, refresh=True, **kwargs):
        pass


@contextmanager
def stage_progress(desc, total, *, unit="batch"):
    # Passing desc=None means silent internal evaluation, e.g. epoch validation.
    active = bool(desc and _enabled)
    if desc and _configured and not active:
        console_info("  %s (%d %s)", desc, total, unit)
    if not active:
        yield QuietProgress()
        return
    with tqdm(total=total, desc=desc, unit=unit,
              file=_stream, leave=False, dynamic_ncols=True,
              ascii=True, mininterval=.5) as bar:
        yield bar
