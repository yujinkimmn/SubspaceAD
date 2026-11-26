import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import csam_progress as pr


class TTYStream(io.StringIO):
    def isatty(self):
        return True


class FakeBar:
    """Exercise lifecycle and logging integration even without the package installed."""
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.n = 0
        self.closed = False
        self.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def update(self, n=1):
        self.n += n

    def set_description_str(self, desc, refresh=True):
        self.kwargs["desc"] = desc

    def set_postfix(self, refresh=True, **kwargs):
        pass

    @staticmethod
    def write(message, file):
        file.write(message+"\n")


class TestCompactProgress(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)

    def tearDown(self):
        for handler in list(pr.LOG.handlers):
            pr.LOG.removeHandler(handler)
            handler.close()
        pr.LOG.propagate=True
        pr._configured=False
        pr._enabled=False
        pr._stream=None
        self.tmp.cleanup()

    def test_auto_redirected_output_is_compact_and_full_log_is_retained(self):
        stream=io.StringIO()
        pr.setup_logging(self.root,stream=stream)
        pr.LOG.info("[Batch] detail hidden")
        pr.LOG.info("[Memory] detail hidden")
        pr.console_info("[Experiment 1/6] P0")
        pr.LOG.warning("skipped warning")
        with pr.stage_progress("P0 Gaussian fit",4,unit="img") as bar:
            bar.update(4)
        pr.epoch_note("epoch %d loss=%.3f",1,.1)
        with pr.stage_progress(None,4) as bar:
            bar.update(4)
        console=stream.getvalue()
        log=(self.root/"run.log").read_text(encoding="utf-8")
        for token in ("[Experiment 1/6] P0","skipped warning","P0 Gaussian fit (4 img)","epoch 1 loss=0.100"):
            self.assertIn(token,console)
            self.assertIn(token,log)
        self.assertNotIn("[Batch]",console)
        self.assertNotIn("[Memory]",console)
        self.assertIn("[Batch] detail hidden",log)
        self.assertIn("[Memory] detail hidden",log)
        self.assertNotIn("\r",console)
        self.assertNotIn("\r",log)

    def test_verbose_keeps_the_old_detailed_console(self):
        stream=io.StringIO()
        pr.setup_logging(self.root,verbose=True,progress_mode="off",stream=stream)
        pr.LOG.info("[Batch] visible in verbose mode")
        self.assertIn("[Batch] visible in verbose mode",stream.getvalue())

    def test_auto_tty_progress_and_close_on_failure(self):
        stream=TTYStream()
        FakeBar.instances.clear()
        with patch.object(pr,"tqdm",FakeBar):
            pr.setup_logging(self.root,stream=stream)
            with self.assertRaisesRegex(RuntimeError,"boom"):
                with pr.stage_progress("P2 train",5) as bar:
                    bar.update(2)
                    bar.set_description_str("P2 validation")
                    pr.console_info("important result")
                    pr.epoch_note("suppressed while bar active")
                    raise RuntimeError("boom")
            active=FakeBar.instances[0]
            self.assertTrue(active.closed)
            self.assertEqual(active.n,2)
            self.assertEqual(active.kwargs["total"],5)
            self.assertFalse(active.kwargs["leave"])
            self.assertIn("important result",stream.getvalue())
            self.assertNotIn("suppressed",stream.getvalue())
            with pr.stage_progress(None,3) as bar:
                bar.update(3)
            self.assertEqual(len(FakeBar.instances),1)

    def test_missing_tqdm_falls_back_without_breaking_offline_runs(self):
        stream=io.StringIO()
        with patch.object(pr,"tqdm",None):
            pr.setup_logging(self.root,progress_mode="on",stream=stream)
            with pr.stage_progress("P0 validation",10,unit="img") as bar:
                bar.update(10)
        self.assertIn("tqdm is not installed",stream.getvalue())
        self.assertIn("P0 validation (10 img)",stream.getvalue())

    @unittest.skipIf(pr.tqdm is None,"tqdm not installed in this environment")
    def test_real_tqdm_force_on_preserves_plain_file_log(self):
        stream=io.StringIO()
        pr.setup_logging(self.root,progress_mode="on",stream=stream)
        with pr.stage_progress("P5 train",3) as bar:
            bar.set_postfix(loss="0.15",refresh=False)
            bar.update(3)
            bar.refresh()
            pr.console_info("summary")
        self.assertIn("P5 train",stream.getvalue())
        self.assertIn("summary",stream.getvalue())
        log=(self.root/"run.log").read_text(encoding="utf-8")
        self.assertIn("summary",log)
        self.assertNotIn("P5 train",log)


if __name__=="__main__":
    unittest.main()
