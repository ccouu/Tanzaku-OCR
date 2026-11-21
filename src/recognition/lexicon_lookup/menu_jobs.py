"""Bounded single-worker asynchronous job manager for full-menu inspection.

Design invariants:
- Single dedicated worker thread + bounded FIFO queue (no thread-per-crop explosion).
- Bounded resident job store (`max_resident_jobs`) evicting oldest finished jobs.
- Unique `job_id` and `config_signature` per submission so config changes never
  mix results within a menu and obsolete jobs never overwrite a newer UI photo.
- Lightweight status/result/crop-detail reads hit in-memory cache only (never
  load or re-run models per poll).
"""
from __future__ import annotations

import queue
import sys
import threading
import time
import traceback
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional

from .contracts import LookupConfig
from .menu_pipeline import (
    MAX_CROPS_PER_MENU,
    make_config_signature,
    run_full_menu_pipeline,
)
from .pipeline import LexiconPipeline


@dataclass
class _MenuJobRecord:
    job_id: str
    filename: str
    decode_mode: str
    lookup_config: LookupConfig
    confidence: float
    config_signature: str
    created_utc: float
    image_bytes: Optional[bytes]
    status: str = "queued"  # queued | running | completed | failed | cancelled
    stage: str = "queued"
    crops_done: int = 0
    crops_total: int = 0
    error: Optional[str] = None
    overview: Optional[dict[str, Any]] = None
    crop_details: dict[str, dict[str, Any]] = field(default_factory=dict)
    cancel_event: threading.Event = field(default_factory=threading.Event)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None


class MenuJobManager:
    """Thread-safe bounded async executor for full-menu inspection jobs."""

    def __init__(
        self,
        *,
        detector: Any,
        pipeline: LexiconPipeline,
        max_resident_jobs: int = 6,
        max_queue_size: int = 8,
        max_crops: int = MAX_CROPS_PER_MENU,
    ) -> None:
        if max_resident_jobs < 1 or max_queue_size < 1:
            raise ValueError("max_resident_jobs and max_queue_size must be >= 1")
        self._detector = detector
        self._pipeline = pipeline
        self._max_resident = int(max_resident_jobs)
        self._max_crops = int(max_crops)
        self._lock = threading.Lock()
        self._jobs: OrderedDict[str, _MenuJobRecord] = OrderedDict()
        self._queue: queue.Queue[Optional[str]] = queue.Queue(maxsize=int(max_queue_size))
        self._shutdown = False
        self._worker = threading.Thread(target=self._worker_loop, name="menu-job-worker", daemon=True)
        self._worker.start()

    def close(self) -> None:
        self._shutdown = True
        with self._lock:
            for rec in self._jobs.values():
                rec.cancel_event.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        self._worker.join(timeout=2.0)

    def _evict_if_needed_locked(self) -> None:
        while len(self._jobs) > self._max_resident:
            # Prefer evicting oldest terminal job first
            victim_id = None
            for jid, rec in self._jobs.items():
                if rec.status in ("completed", "failed", "cancelled"):
                    victim_id = jid
                    break
            if victim_id is None:
                break
            del self._jobs[victim_id]

    def submit(
        self,
        *,
        image_bytes: bytes,
        filename: str,
        decode_mode: str,
        lookup_config: LookupConfig,
        confidence: float = 0.35,
    ) -> dict[str, Any]:
        if self._detector is None:
            raise RuntimeError("No full-menu YOLO detector attached to this server instance")
        if not self._pipeline.has_model:
            raise RuntimeError("No recognizer model attached to this server instance")
        if not image_bytes:
            raise ValueError("image_bytes must be non-empty")

        jid = uuid.uuid4().hex[:16]
        sig = make_config_signature(
            decode_mode=decode_mode,
            lookup_config=lookup_config,
            confidence=confidence,
        )
        rec = _MenuJobRecord(
            job_id=jid,
            filename=filename,
            decode_mode=decode_mode,
            lookup_config=lookup_config,
            confidence=float(confidence),
            config_signature=sig,
            created_utc=time.time(),
            image_bytes=image_bytes,
        )
        with self._lock:
            self._jobs[jid] = rec
            self._evict_if_needed_locked()
        try:
            self._queue.put_nowait(jid)
        except queue.Full as exc:
            with self._lock:
                self._jobs.pop(jid, None)
            raise OverflowError("Menu job queue is full; please wait for active jobs to finish.") from exc
        return self.get_status(jid)

    def rerun_from_existing(
        self,
        source_job_id: str,
        *,
        decode_mode: str,
        lookup_config: LookupConfig,
        confidence: float = 0.35,
    ) -> dict[str, Any]:
        with self._lock:
            src = self._jobs.get(source_job_id)
            if src is None:
                raise KeyError(f"Unknown source job_id: {source_job_id!r}")
            if not src.image_bytes:
                raise ValueError("Source job image bytes are no longer resident; please re-upload photo.")
            img_bytes = src.image_bytes
            fname = src.filename
        return self.submit(
            image_bytes=img_bytes,
            filename=fname,
            decode_mode=decode_mode,
            lookup_config=lookup_config,
            confidence=confidence,
        )

    def cancel(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            rec = self._jobs.get(job_id)
            if rec is None:
                raise KeyError(f"Unknown job_id: {job_id!r}")
            rec.cancel_event.set()
            if rec.status == "queued":
                rec.status = "cancelled"
                rec.stage = "cancelled"
                rec.finished_at = time.time()
        return self.get_status(job_id)

    def get_status(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            rec = self._jobs.get(job_id)
            if rec is None:
                raise KeyError(f"Unknown job_id: {job_id!r}")
            return {
                "job_id": rec.job_id,
                "filename": rec.filename,
                "status": rec.status,
                "stage": rec.stage,
                "crops_done": rec.crops_done,
                "crops_total": rec.crops_total,
                "decode_mode": rec.decode_mode,
                "confidence": rec.confidence,
                "config_signature": rec.config_signature,
                "error": rec.error,
                "counts": rec.overview["counts"] if rec.overview else None,
                "timings_ms": rec.overview["timings_ms"] if rec.overview else None,
            }

    def get_result(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            rec = self._jobs.get(job_id)
            if rec is None:
                raise KeyError(f"Unknown job_id: {job_id!r}")
            if rec.status != "completed" or rec.overview is None:
                raise ValueError(f"Job {job_id!r} is not completed (status={rec.status!r})")
            return {
                "job_id": rec.job_id,
                "status": rec.status,
                "overview": rec.overview,
            }

    def get_crop_detail(self, job_id: str, crop_id: str) -> dict[str, Any]:
        with self._lock:
            rec = self._jobs.get(job_id)
            if rec is None:
                raise KeyError(f"Unknown job_id: {job_id!r}")
            if rec.status != "completed":
                raise ValueError(f"Job {job_id!r} is not completed (status={rec.status!r})")
            detail = rec.crop_details.get(crop_id)
            if detail is None:
                raise KeyError(f"Unknown crop_id {crop_id!r} for job {job_id!r}")
            return {"job_id": rec.job_id, "crop_detail": detail}

    def _worker_loop(self) -> None:
        while not self._shutdown:
            jid = self._queue.get()
            if jid is None:
                break
            with self._lock:
                rec = self._jobs.get(jid)
                if rec is None or rec.cancel_event.is_set() or rec.status == "cancelled":
                    if rec is not None and rec.status != "cancelled":
                        rec.status = "cancelled"
                        rec.stage = "cancelled"
                        rec.finished_at = time.time()
                    self._queue.task_done()
                    continue
                rec.status = "running"
                rec.stage = "starting"
                rec.started_at = time.time()
                img_bytes = rec.image_bytes
                fname = rec.filename
                d_mode = rec.decode_mode
                l_cfg = rec.lookup_config
                conf = rec.confidence
                cancel_ev = rec.cancel_event

            def _on_progress(stage: str, done: int, total: int) -> None:
                with self._lock:
                    r = self._jobs.get(jid)
                    if r is not None:
                        r.stage = stage
                        r.crops_done = int(done)
                        r.crops_total = int(total)

            try:
                assert img_bytes is not None
                out = run_full_menu_pipeline(
                    image_bytes=img_bytes,
                    filename=fname,
                    detector=self._detector,
                    pipeline=self._pipeline,
                    lookup_config=l_cfg,
                    decode_mode=d_mode,
                    confidence=conf,
                    max_crops=self._max_crops,
                    progress_callback=_on_progress,
                    cancel_event=cancel_ev,
                )
                with self._lock:
                    r = self._jobs.get(jid)
                    if r is not None:
                        r.overview = out["overview"]
                        r.crop_details = out["crop_details"]
                        r.status = "completed"
                        r.stage = "completed"
                        r.finished_at = time.time()
                sys.stdout.write(
                    f"[menu_job] completed job={jid} crops={out['overview']['counts']} "
                    f"total_ms={out['overview']['timings_ms']['total_ms']}\n"
                )
                sys.stdout.flush()
            except InterruptedError:
                with self._lock:
                    r = self._jobs.get(jid)
                    if r is not None:
                        r.status = "cancelled"
                        r.stage = "cancelled"
                        r.finished_at = time.time()
            except Exception as exc:
                err_txt = f"{type(exc).__name__}: {exc}"
                sys.stderr.write(f"[menu_job] failed job={jid}: {err_txt}\n")
                traceback.print_exc(file=sys.stderr)
                sys.stderr.flush()
                with self._lock:
                    r = self._jobs.get(jid)
                    if r is not None:
                        r.status = "failed"
                        r.stage = "failed"
                        r.error = err_txt
                        r.finished_at = time.time()
            finally:
                self._queue.task_done()
