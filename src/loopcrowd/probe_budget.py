"""Cooperative synchronized phase accounting with injectable device callbacks."""

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
        self.refusals = []

    def _check(self, prospective_seconds):
        if self.seconds + prospective_seconds > self.seconds_limit:
            raise RuntimeError("diagnostic GPU-phase budget would be exceeded")
        if self.peak_bytes > self.bytes_limit:
            raise RuntimeError("diagnostic process reserved memory exceeded the cap")

    def _crossings(self):
        return [label for label, actual, limit in (
            ("gpu_seconds", self.seconds, self.seconds_limit),
            ("process_peak_reserved_bytes", self.peak_bytes, self.bytes_limit),
            ("cases", self.cases, self.case_limit),
        ) if actual > limit]

    @contextmanager
    def phase(self, name, *, estimate_seconds, batch_policy_case=False, cleanup=False):
        """Measure entered phases, including mandatory release after an abort."""
        if not isfinite(estimate_seconds) or estimate_seconds <= 0:
            raise ValueError("a phase needs a positive finite estimate")
        if cleanup and batch_policy_case:
            raise ValueError("mandatory cleanup cannot enter a measurement case")
        if not cleanup:
            try:
                self.peak_bytes = max(self.peak_bytes, self.peak_reserved())
                self._check(estimate_seconds)
                if batch_policy_case and self.cases >= self.case_limit:
                    raise RuntimeError("diagnostic batch-policy case cap reached")
            except Exception as error:
                self.refusals.append({"name": name, "estimate_seconds": estimate_seconds,
                                      "cumulative_seconds": self.seconds, "peak_bytes": self.peak_bytes,
                                      "cases_entered": self.cases, "reason": str(error),
                                      "budget_crossings": self._crossings()})
                raise
        started = self.clock()
        status = "failed"
        sync_error = None
        try:
            if not cleanup:
                self.synchronize()
            self.cases += int(batch_policy_case)
            yield
            status = "completed"
        finally:
            try:
                self.synchronize()
            except Exception as error:
                sync_error = str(error)
                status = "synchronization_or_allocator_check_failed"
                raise
            finally:
                # A failed final sync must propagate. The host allocator query
                # does not depend on the device successfully finishing work.
                try:
                    self.peak_bytes = max(self.peak_bytes, self.peak_reserved())
                except Exception as error:
                    status = "synchronization_or_allocator_check_failed"
                    if sync_error is None:
                        raise
                finally:
                    elapsed = self.clock() - started
                    self.seconds += elapsed
                    crossings = self._crossings()
                    if status == "completed" and crossings:
                        status = "budget_exceeded"
                    self.phases.append({"name": name, "status": status, "elapsed_seconds": elapsed,
                                        "cumulative_seconds": self.seconds, "peak_bytes": self.peak_bytes,
                                        "cases_entered": self.cases, "budget_crossings": crossings,
                                        "final_sync_error": sync_error})
        self._check(0)
