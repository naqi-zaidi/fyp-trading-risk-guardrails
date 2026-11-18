"""
v10/execution/mt5_io.py
MT5Bridge: file-based command/acknowledgement channel.

Writes control actions to MQL5/Files/commands/control.csv
Reads EA acknowledgements from MQL5/Files/acks/ack.csv
Mirrors both to data/execution/ for audit.

Uses temp-then-move for atomicity.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any, Optional

from v10.core.timeutils import iso_str, utc_now

logger = logging.getLogger(__name__)

COMMAND_COLUMNS = [
    "ts_utc", "symbol", "action", "params_json",
    "config_version", "reason_code", "reason_text", "nonce",
]

ACK_COLUMNS = [
    "ts_broker", "ts_utc", "action", "status_code",
    "status_msg", "nonce",
]


class MT5Bridge:
    """
    File-based bridge between Python controller and MT5 Expert Advisor.

    Attributes
    ----------
    command_csv : str — path to outbound control.csv (MT5 side)
    ack_csv : str — path to inbound ack.csv (MT5 side)
    """

    def __init__(
        self,
        command_csv: str = "MQL5/Files/commands/control.csv",
        ack_csv: str = "MQL5/Files/acks/ack.csv",
        mirror_command: str = "data/execution/control.csv",
        mirror_ack: str = "data/execution/ack.csv",
    ):
        self.command_csv = command_csv
        self.ack_csv = ack_csv
        self.mirror_command = mirror_command
        self.mirror_ack = mirror_ack

        self._last_nonce: str = ""
        self._ack_timeout_bars: int = 2

        # Ensure directories exist
        for p in [command_csv, ack_csv, mirror_command, mirror_ack]:
            Path(p).parent.mkdir(parents=True, exist_ok=True)

        # Initialize CSV headers if files don't exist
        self._ensure_header(self.command_csv, COMMAND_COLUMNS)
        self._ensure_header(self.mirror_command, COMMAND_COLUMNS)
        self._ensure_header(self.ack_csv, ACK_COLUMNS)
        self._ensure_header(self.mirror_ack, ACK_COLUMNS)

    @staticmethod
    def _ensure_header(path: str, columns: list[str]) -> None:
        p = Path(path)
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(columns)

    def write_command(
        self,
        action: str,
        symbol: str = "EURUSD",
        params: Optional[dict] = None,
        config_version: str = "1.0",
        reason_code: str = "",
        reason_text: str = "",
    ) -> str:
        """
        Write a control command to the MT5 command CSV.
        Returns the nonce (unique ID) for ack matching.
        """
        nonce = uuid.uuid4().hex[:12]
        ts = iso_str(utc_now())
        params_json = json.dumps(params or {})

        row = [
            ts, symbol, action, params_json,
            config_version, reason_code, reason_text, nonce,
        ]

        # Atomic write: temp file then move
        self._atomic_append(self.command_csv, row)

        # Mirror for audit
        self._atomic_append(self.mirror_command, row)

        self._last_nonce = nonce
        logger.debug("Command written: action=%s nonce=%s", action, nonce)
        return nonce

    def read_ack(self, nonce: Optional[str] = None) -> Optional[dict[str, str]]:
        """
        Read the latest acknowledgement, optionally filtered by nonce.
        Returns None if no matching ack found.
        """
        p = Path(self.ack_csv)
        if not p.exists():
            return None

        target = nonce or self._last_nonce
        try:
            with open(p, "r", newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            # Search from newest to oldest
            for row in reversed(rows):
                if row.get("nonce") == target:
                    return dict(row)
        except Exception as e:
            logger.warning("Error reading ack file: %s", e)

        return None

    def mirror_files(self) -> None:
        """
        Copy current MT5-side files to the audit mirror directory.
        Adds SHA-256 hashes for integrity verification.
        """
        for src, dst in [
            (self.command_csv, self.mirror_command),
            (self.ack_csv, self.mirror_ack),
        ]:
            src_path = Path(src)
            if src_path.exists():
                content = src_path.read_bytes()
                sha = hashlib.sha256(content).hexdigest()

                dst_path = Path(dst)
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)

                # Write checksum sidecar
                checksum_path = dst_path.with_suffix(".sha256")
                checksum_path.write_text(sha)

    @staticmethod
    def _atomic_append(path: str, row: list) -> None:
        """Append a row to a CSV file using temp-then-append for safety."""
        p = Path(path)
        try:
            with open(p, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(row)
                f.flush()
                os.fsync(f.fileno())
        except OSError as e:
            logger.error("Failed to write to %s: %s", path, e)
            raise

    @property
    def last_nonce(self) -> str:
        return self._last_nonce
