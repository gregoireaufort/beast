"""Lightweight BEAST runtime profiling helpers."""

import atexit
from contextlib import contextmanager
import os
import resource
import sys
import time

_PROFILE_EVENTS = []
_SUMMARY_PRINTED = False


def profiling_enabled():
    return os.environ.get("BEAST_PROFILE", "").lower() in {"1", "true", "yes", "on"}


def _rss_mb():
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return usage / (1024.0 * 1024.0)
    return usage / 1024.0


@contextmanager
def profile_stage(stage, detail=None, log_event=False):
    if not profiling_enabled():
        yield
        return

    rss_start = _rss_mb()
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        rss_end = _rss_mb()
        event = {
            "stage": stage,
            "detail": detail,
            "seconds": elapsed,
            "rss_start_mb": rss_start,
            "rss_end_mb": rss_end,
            "rss_delta_mb": rss_end - rss_start,
        }
        _PROFILE_EVENTS.append(event)
        if log_event or os.environ.get("BEAST_PROFILE_LOG_EVENTS", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            label = stage if detail is None else f"{stage}: {detail}"
            print(
                "[BEAST_PROFILE] "
                f"{label} wall={elapsed:.6f}s "
                f"rss={rss_end:.1f}MB delta={event['rss_delta_mb']:.1f}MB"
            )


def profile_summary(label="BEAST profile summary", reset=False):
    global _PROFILE_EVENTS, _SUMMARY_PRINTED

    if not profiling_enabled() or len(_PROFILE_EVENTS) == 0:
        return {}

    totals = {}
    peaks = {}
    for event in _PROFILE_EVENTS:
        stage = event["stage"]
        totals[stage] = totals.get(stage, 0.0) + event["seconds"]
        peaks[stage] = max(peaks.get(stage, 0.0), event["rss_end_mb"])

    total_seconds = sum(totals.values())
    print(f"[BEAST_PROFILE] {label}")
    print("[BEAST_PROFILE] stage,seconds,percent,peak_rss_mb")
    for stage, seconds in sorted(totals.items(), key=lambda item: item[1], reverse=True):
        percent = 100.0 * seconds / total_seconds if total_seconds > 0.0 else 0.0
        print(
            f"[BEAST_PROFILE] {stage},{seconds:.6f},"
            f"{percent:.2f},{peaks[stage]:.1f}"
        )
    print(f"[BEAST_PROFILE] total,{total_seconds:.6f},100.00,{_rss_mb():.1f}")

    if reset:
        _PROFILE_EVENTS = []

    _SUMMARY_PRINTED = True
    return totals


def _atexit_summary():
    global _SUMMARY_PRINTED
    if profiling_enabled() and not _SUMMARY_PRINTED:
        _SUMMARY_PRINTED = True
        profile_summary("BEAST process profile summary")


atexit.register(_atexit_summary)
