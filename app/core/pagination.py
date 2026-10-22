import base64
import json
from datetime import datetime, timezone
from typing import Tuple

from app.core.errors import AppError


def utcnow() -> datetime:
    """Return current UTC time rounded down to whole milliseconds."""
    dt = datetime.now(timezone.utc)
    return dt.replace(microsecond=(dt.microsecond // 1000) * 1000)


def encode_cursor(updated_at: datetime, _id: str) -> str:
    """Encode updated_at and id into a URL-safe base64 string."""
    iso_str = updated_at.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    json_bytes = json.dumps([iso_str, _id]).encode("utf-8")
    return base64.urlsafe_b64encode(json_bytes).decode("utf-8").rstrip("=")


def decode_cursor(text: str) -> Tuple[datetime, str]:
    """Decode cursor back to (updated_at, id). Raises INVALID_CURSOR on failure."""
    try:
        padding = "=" * ((4 - len(text) % 4) % 4)
        json_bytes = base64.urlsafe_b64decode(text + padding)
        data = json.loads(json_bytes.decode("utf-8"))
        if not isinstance(data, list) or len(data) != 2:
            raise ValueError()

        iso_str, _id = data
        if not isinstance(iso_str, str) or not isinstance(_id, str) or len(_id) != 32:
            raise ValueError()

        dt = datetime.strptime(iso_str, "%Y-%m-%dT%H:%M:%S.%fZ").replace(
            tzinfo=timezone.utc
        )
        return dt, _id
    except Exception:
        raise AppError.from_code("INVALID_CURSOR")
