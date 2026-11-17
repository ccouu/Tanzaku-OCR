"""Reusable Colab A100 remote lifecycle manager (zero-LLM-polling, chunked upload, job-id ownership, SHA-256 sync, herdr alerts)."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile
import time
import uuid

from src.recognition.checkpointing import sha256_file


@dataclass
class ColabJobConfig:
    stage: str
    session: str
    stage_dir: Path
    source_ckpt: Path
    ckpt_subdirs: tuple[str, ...]
    remote_entrypoint: str = "-m src.recognition.trainer_v2"
    remote_args: list[str] = field(default_factory=list)
    gpu: str = "A100"
    poll_interval_s: int = 45
    stall_timeout_s: int = 600
    chunk_size_bytes: int = 20 * 1024 * 1024
    herdr_targets: tuple[str, ...] | None = None
    job_id: str | None = None
    bundle_dirs: tuple[str, ...] = ("code", "payload", "images")
    expected_checkpoints: tuple[str, ...] = ()
    extra_downloads: tuple[str, ...] = ()
    stop_on_done: bool = True


def discover_agy_panes() -> list[str]:
    """Query `herdr pane list` and return pane_ids where agent == 'agy'."""
    try:
        cp = subprocess.run(
            ["herdr", "pane", "list"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if cp.returncode == 0 and cp.stdout.strip():
            data = json.loads(cp.stdout)
            panes = data.get("result", {}).get("panes", [])
            agy_panes = [p["pane_id"] for p in panes if p.get("agent") == "agy" and p.get("pane_id")]
            if agy_panes:
                return agy_panes
    except Exception:
        pass
    pane_env = os.environ.get("HERDR_PANE_ID")
    return [pane_env] if pane_env else []


def notify_herdr(msg: str, targets: tuple[str, ...] | None = None) -> None:
    """Send prompt notification to agent=='agy' panes via `herdr agent prompt` (fire-and-forget)."""
    all_targets = list(targets) if targets is not None else discover_agy_panes()
    for tgt in all_targets:
        try:
            subprocess.run(
                ["herdr", "agent", "prompt", tgt, msg],
                check=False,
                capture_output=True,
                text=True,
                timeout=15,
            )
        except Exception:
            pass


def _run_cmd(cmd: list[str], check: bool = True, timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=check, capture_output=True, text=True, timeout=timeout)


def colab_exec_file(session: str, inline_py: str, work_dir: Path, retries: int = 5, timeout: int = 120) -> str:
    work_dir.mkdir(parents=True, exist_ok=True)
    tmp_py = work_dir / f"_exec_{os.getpid()}.py"
    tmp_py.write_text(inline_py, encoding="utf-8", newline="\n")
    last_err = None
    for attempt in range(retries):
        try:
            cp = _run_cmd(
                ["colab", "exec", "-s", session, "-f", str(tmp_py), "--timeout", str(float(timeout))],
                check=True,
                timeout=timeout + 30,
            )
            return cp.stdout
        except Exception as exc:
            last_err = exc
            time.sleep(min(30, 5 * (attempt + 1)))
    raise RuntimeError(f"colab exec failed on {session} after {retries} attempts: {last_err}")


def upload_chunked(
    session: str,
    local_file: Path,
    remote_file: str,
    work_dir: Path,
    chunk_size: int = 20 * 1024 * 1024,
) -> None:
    part_dir = work_dir / f"{local_file.name}_parts"
    if part_dir.exists():
        shutil.rmtree(part_dir)
    part_dir.mkdir(parents=True, exist_ok=True)
    part_names: list[str] = []
    with local_file.open("rb") as f:
        idx = 0
        while True:
            buf = f.read(chunk_size)
            if not buf:
                break
            pname = f"{local_file.name}.part{idx:02d}"
            ppath = part_dir / pname
            ppath.write_bytes(buf)
            part_names.append(pname)
            remote_part = f"{remote_file}.part{idx:02d}"
            last_err = None
            for attempt in range(4):
                try:
                    _run_cmd(["colab", "upload", "-s", session, str(ppath), remote_part], check=True, timeout=300)
                    break
                except Exception as exc:
                    last_err = exc
                    time.sleep(5 * (attempt + 1))
            else:
                raise RuntimeError(f"Chunk upload failed for {remote_part}: {last_err}")
            idx += 1
    reasm_py = (
        "import pathlib\n"
        f"dst = pathlib.Path('{remote_file}')\n"
        "dst.parent.mkdir(parents=True, exist_ok=True)\n"
        f"parts = {part_names!r}\n"
        "with dst.open('wb') as out:\n"
        "    for pn in parts:\n"
        "        p = dst.parent / pn\n"
        "        out.write(p.read_bytes())\n"
        "        p.unlink()\n"
    )
    colab_exec_file(session, reasm_py, work_dir, timeout=180)
    shutil.rmtree(part_dir, ignore_errors=True)


def colab_download_verified(
    session: str,
    remote_path: str,
    local_path: Path,
    expected_sha: str | None = None,
    retries: int = 4,
) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if local_path.exists() and expected_sha and sha256_file(local_path) == expected_sha:
        return
    tmp = local_path.with_suffix(local_path.suffix + ".part")
    last_err = None
    for attempt in range(retries):
        try:
            if tmp.exists():
                tmp.unlink()
            _run_cmd(["colab", "download", "-s", session, remote_path, str(tmp)], check=True, timeout=600)
            if expected_sha:
                got = sha256_file(tmp)
                if got != expected_sha:
                    raise RuntimeError(f"SHA mismatch for {remote_path}: expected {expected_sha}, got {got}")
            tmp.replace(local_path)
            return
        except Exception as exc:
            last_err = exc
            time.sleep(min(30, 5 * (attempt + 1)))
    raise RuntimeError(f"Failed downloading {remote_path} from {session}: {last_err}")


def make_bundle_tar(
    stage_dir: Path,
    source_ckpt: Path,
    bundle_path: Path,
    include_src_recognition: bool = True,
    bundle_dirs: tuple[str, ...] = ("code", "payload", "images"),
) -> None:
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(bundle_path, "w:gz") as tf:
        for sub in bundle_dirs:
            src = stage_dir / sub
            if not src.exists():
                continue
            for p in sorted(src.rglob("*")):
                if not p.is_file() or "__pycache__" in p.parts or p.suffix == ".pyc":
                    continue
                tf.add(p, arcname=str(Path(sub) / p.relative_to(src)))
        if include_src_recognition:
            src_rec = Path(__file__).resolve().parent
            for p in sorted(src_rec.rglob("*")):
                if not p.is_file() or "__pycache__" in p.parts or p.suffix in (".pyc", ".nbc", ".nbi"):
                    continue
                tf.add(p, arcname=str(Path("src") / "recognition" / p.relative_to(src_rec)))
        tf.add(source_ckpt, arcname="payload/source_ckpt.pt")


def build_probe_script(remote_root: str, job_id: str, ckpt_subdirs: list[str]) -> str:
    """Build remote probe script that verifies job_id ownership on DONE.json and exact PID liveness."""
    return (
        "import json, os, pathlib\n"
        f"root = pathlib.Path({remote_root!r})\n"
        f"job_id = {job_id!r}\n"
        "log = root / 'remote.log'\n"
        "txt = log.read_text(encoding='utf-8', errors='replace') if log.exists() else ''\n"
        "pid_file = root / f'job_{job_id}.pid'\n"
        "pid = pid_file.read_text().strip() if pid_file.exists() else ''\n"
        "alive = False\n"
        "if pid.isdigit():\n"
        "    try:\n"
        "        os.kill(int(pid), 0)\n"
        "        st_p = pathlib.Path(f'/proc/{pid}/status')\n"
        "        if st_p.exists():\n"
        "            st_txt = st_p.read_text(encoding='utf-8', errors='replace')\n"
        "            is_zombie = any(l.startswith('State:') and 'Z' in l for l in st_txt.splitlines())\n"
        "            alive = not is_zombie\n"
        "        else:\n"
        "            alive = False\n"
        "    except OSError:\n"
        "        alive = False\n"
        "done_path = root / 'DONE.json'\n"
        "done_owned = False\n"
        "done_payload = None\n"
        "if done_path.exists():\n"
        "    try:\n"
        "        done_payload = json.loads(done_path.read_text(encoding='utf-8'))\n"
        "        done_owned = (done_payload.get('job_id') == job_id)\n"
        "    except Exception:\n"
        "        done_owned = False\n"
        "cksums = {}\n"
        f"for sub in {list(ckpt_subdirs)!r}:\n"
        "    p = root / 'checkpoints' / sub / 'checksums.json'\n"
        "    if p.exists():\n"
        "        try:\n"
        "            cksums[sub] = json.loads(p.read_text(encoding='utf-8'))\n"
        "        except Exception:\n"
        "            pass\n"
        "print(json.dumps({\n"
        "    'log_size': len(txt),\n"
        "    'tail': txt.splitlines()[-15:],\n"
        "    'alive': alive,\n"
        "    'pid': pid,\n"
        "    'done': done_owned,\n"
        "    'checksums': cksums,\n"
        "}))\n"
    )


def run_colab_job(cfg: ColabJobConfig) -> dict:
    """Provision/reuse Colab runtime, upload bundle, launch job-owned process, sync SHA-256 artifacts, and stop."""
    job_id = cfg.job_id or f"{cfg.stage}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    remote_root = f"/content/{cfg.stage}_run"
    tmp_exec_dir = cfg.stage_dir / "chunks"

    ls_cp = _run_cmd(["colab", "ls"], check=False, timeout=30)
    ls_out = (ls_cp.stdout or "") + "\n" + (ls_cp.stderr or "")
    if cfg.session not in ls_out:
        _run_cmd(["colab", "new", "--gpu", cfg.gpu, "-s", cfg.session], check=True, timeout=300)

    # Clear stale DONE.json / remote.log / old pid markers before launching this job_id
    prep_py = (
        "import pathlib\n"
        f"root = pathlib.Path({remote_root!r})\n"
        "root.mkdir(parents=True, exist_ok=True)\n"
        "for name in ('DONE.json', 'remote.log'):\n"
        "    p = root / name\n"
        "    if p.exists():\n"
        "        p.unlink()\n"
        "for p in root.glob('job_*.pid'):\n"
        "    p.unlink()\n"
    )
    colab_exec_file(cfg.session, prep_py, tmp_exec_dir)

    bundle_path = cfg.stage_dir / "chunks" / f"{cfg.stage}_bundle.tar.gz"
    make_bundle_tar(cfg.stage_dir, cfg.source_ckpt, bundle_path, bundle_dirs=cfg.bundle_dirs)
    upload_chunked(cfg.session, bundle_path, f"{remote_root}/{cfg.stage}_bundle.tar.gz", tmp_exec_dir, cfg.chunk_size_bytes)
    colab_exec_file(
        cfg.session,
        f"import tarfile; tf=tarfile.open('{remote_root}/{cfg.stage}_bundle.tar.gz'); tf.extractall('{remote_root}'); tf.close()",
        tmp_exec_dir,
    )

    if cfg.remote_entrypoint.startswith("-m "):
        ep_parts = shlex.split(cfg.remote_entrypoint)
    else:
        ep_parts = [f"{remote_root}/{cfg.remote_entrypoint}"]
    py_cmd = " ".join(
        shlex.quote(x) for x in ["python3", "-u", *ep_parts, *cfg.remote_args, "--job-id", job_id]
    )
    bash_cmd = (
        f"cd {shlex.quote(remote_root)} && "
        f"echo $$ > {shlex.quote(remote_root + f'/job_{job_id}.pid')} && "
        f"exec env PYTHONPATH={shlex.quote(remote_root)} {py_cmd}"
    )
    launch_py = (
        "import subprocess\n"
        f"cmd = {bash_cmd!r} + ' > ' + {remote_root + '/remote.log'!r} + ' 2>&1'\n"
        "subprocess.Popen(['nohup', 'bash', '-lc', cmd], start_new_session=True)\n"
    )
    colab_exec_file(cfg.session, launch_py, tmp_exec_dir)

    ack_sent = False
    last_log_size = -1
    last_growth_ts = time.time()
    downloaded: dict[str, str] = {}
    probe_py = build_probe_script(remote_root, job_id, list(cfg.ckpt_subdirs))

    while True:
        time.sleep(cfg.poll_interval_s)
        raw = colab_exec_file(cfg.session, probe_py, tmp_exec_dir)
        state = json.loads(raw.strip().splitlines()[-1])
        (cfg.stage_dir / "logs").mkdir(parents=True, exist_ok=True)
        (cfg.stage_dir / "logs" / "remote_tail.log").write_text("\n".join(state["tail"]) + "\n", encoding="utf-8")

        if state["log_size"] > last_log_size:
            last_log_size = state["log_size"]
            last_growth_ts = time.time()

        tail_joined = "\n".join(state["tail"])
        if not ack_sent and ("E2E_SMOKE_OK" in tail_joined or '"step": 1,' in tail_joined):
            pid_str = state["pid"] or "?"
            notify_herdr(
                f"{cfg.stage.upper()}_STAGE_ACK: {cfg.stage} (job_id={job_id}) running on {cfg.session}, PID={pid_str}.",
                cfg.herdr_targets,
            )
            ack_sent = True

        for sub, files_map in state["checksums"].items():
            for fname, fsha in files_map.items():
                key = f"{sub}/{fname}"
                if downloaded.get(key) == fsha:
                    continue
                local_dst = cfg.stage_dir / "downloads" / sub / fname
                colab_download_verified(
                    cfg.session,
                    f"{remote_root}/checkpoints/{sub}/{fname}",
                    local_dst,
                    expected_sha=fsha,
                )
                downloaded[key] = fsha

        if state["done"]:
            if cfg.expected_checkpoints:
                missing = [k for k in cfg.expected_checkpoints if k not in downloaded]
                if missing:
                    raise RuntimeError(f"{cfg.stage} DONE.json found but missing expected checkpoints: {missing}")
            colab_download_verified(cfg.session, f"{remote_root}/DONE.json", cfg.stage_dir / "logs" / "DONE.json")
            colab_download_verified(cfg.session, f"{remote_root}/remote.log", cfg.stage_dir / "logs" / "remote.log")
            for sub in cfg.ckpt_subdirs:
                for meta in ("checksums.json", "train_audit.json"):
                    colab_download_verified(
                        cfg.session,
                        f"{remote_root}/checkpoints/{sub}/{meta}",
                        cfg.stage_dir / "downloads" / sub / meta,
                    )
            for rel in cfg.extra_downloads:
                colab_download_verified(
                    cfg.session,
                    f"{remote_root}/{rel}",
                    cfg.stage_dir / "downloads" / rel,
                )
            break

        if not state["alive"] and not state["done"]:
            raise RuntimeError(f"{cfg.stage} (job_id={job_id}) died without owned DONE.json! Tail:\n{tail_joined}")
        if time.time() - last_growth_ts > cfg.stall_timeout_s:
            raise RuntimeError(f"{cfg.stage} (job_id={job_id}) stalled for >{cfg.stall_timeout_s}s! Tail:\n{tail_joined}")

    if cfg.stop_on_done:
        _run_cmd(["colab", "stop", "-s", cfg.session], check=False, timeout=120)
    return {"job_id": job_id, "downloaded": downloaded}
