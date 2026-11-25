"""Cooperative phase accounting with injectable callbacks for CPU-only tests."""

from contextlib import contextmanager
from math import isfinite
from time import perf_counter


class ProbeBudget:
    def __init__(self, *, seconds, reserved_bytes, cases, synchronize, peak_reserved, clock=perf_counter):
        self.seconds_limit, self.bytes_limit, self.case_limit = seconds, reserved_bytes, cases
        self.synchronize, self.peak_reserved, self.clock = synchronize, peak_reserved, clock
        self.seconds = 0.0
        self.peak_bytes = 0
        self.cases = 0
        self.phases = []

    def _check(self, prospective_seconds):
        if self.seconds + prospective_seconds > self.seconds_limit:
            raise RuntimeError("diagnostic GPU-phase budget would be exceeded")
        if self.peak_bytes > self.bytes_limit:
            raise RuntimeError("diagnostic process reserved memory exceeded the cap")

    @contextmanager
    def phase(self, name, *, estimate_seconds, batch_policy_case=False):
        if not isfinite(estimate_seconds) or estimate_seconds <= 0:
            raise ValueError("a phase needs a positive finite estimate")
        self.peak_bytes = max(self.peak_bytes, self.peak_reserved())
        self._check(estimate_seconds)
        if batch_policy_case and self.cases >= self.case_limit:
            raise RuntimeError("diagnostic batch-policy case cap reached")
        started = self.clock()
        status = "failed"
        try:
            self.synchronize()
            self.cases += int(batch_policy_case)
            yield
            status = "completed"
        finally:
            try:
                self.synchronize()
                self.peak_bytes = max(self.peak_bytes, self.peak_reserved())
            except Exception:
                status = "synchronization_or_allocator_check_failed"
                raise
            finally:
                elapsed = self.clock() - started
                self.seconds += elapsed
                self.phases.append({"name": name, "status": status, "elapsed_seconds": elapsed,
                                    "cumulative_seconds": self.seconds, "peak_bytes": self.peak_bytes,
                                    "cases_entered": self.cases})
        self._check(0)
