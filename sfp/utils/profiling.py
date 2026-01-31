import os
import re
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import jax


def _project_root() -> Path:
    return Path(
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"])
        .decode()
        .strip()
    )


def _default_gcs_bucket() -> Optional[str]:
    """Get default GCS bucket from env var or infra/config.local.sh."""
    # Check environment first
    bucket = os.environ.get("GCS_BUCKET")
    if bucket:
        return bucket.removeprefix("gs://")

    # Try parsing config.local.sh
    config_path = _project_root() / "infra" / "config.local.sh"
    if config_path.exists():
        content = config_path.read_text()
        match = re.search(r'GCS_BUCKET=["\']?gs://([^"\'\s]+)', content)
        if match:
            return match.group(1)

    return None


@dataclass
class ProfileConfig:
    traces_dir: Path = field(default_factory=lambda: _project_root() / "traces")
    name: Optional[str] = None
    gcs_bucket: Optional[str] = None
    gcs_prefix: str = "traces"

    def trace_path(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = f"_{self.name}" if self.name else ""
        return self.traces_dir / f"trace_{timestamp}{suffix}"


@contextmanager
def profile(
    config: Optional[ProfileConfig] = None,
    *,
    name: Optional[str] = None,
):
    """
    Context manager for profiling JAX computations.

    Args:
        config: ProfileConfig instance (if None, creates default)
        name: Optional name suffix for the trace directory

    Yields:
        Path to the trace directory

    Example:
        with profile(name="my_kernel") as trace_dir:
            result = my_kernel(inputs)
            result.block_until_ready()
        print(f"Trace saved to: {trace_dir}")
    """
    if config is None:
        config = ProfileConfig(name=name)

    trace_path = config.trace_path()
    trace_path.parent.mkdir(parents=True, exist_ok=True)

    with jax.profiler.trace(str(trace_path)):
        yield trace_path


def upload_to_gcs(
    local_path: Path,
    bucket: Optional[str] = None,
    prefix: str = "traces",
) -> str:
    """
    Upload a trace directory to GCS.

    Args:
        local_path: Path to the local trace directory
        bucket: GCS bucket name (defaults to GCS_BUCKET env var or infra/config.local.sh)
        prefix: Prefix path within the bucket

    Returns:
        GCS URI of the uploaded trace
    """
    if bucket is None:
        bucket = _default_gcs_bucket()
        if bucket is None:
            raise ValueError(
                "No bucket specified and could not find default. "
                "Set GCS_BUCKET env var or add to infra/config.local.sh"
            )

    gcs_path = f"gs://{bucket}/{prefix}/{local_path.name}"
    subprocess.run(
        ["gsutil", "-m", "cp", "-r", str(local_path), gcs_path],
        check=True,
    )
    return gcs_path
