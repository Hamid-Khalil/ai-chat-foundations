import pytest

from app.core.errors import AppError
from app.core.pagination import decode_cursor, encode_cursor, utcnow


def test_pagination_rules():
    now = utcnow()
    assert now.microsecond % 1000 == 0  # rounds to ms
    assert now.tzinfo is not None  # tz-aware

    # Invalid cursors -> 400 INVALID_CURSOR
    with pytest.raises(AppError) as exc:
        decode_cursor("notbase64!")
    assert exc.value.code == "INVALID_CURSOR"
    with pytest.raises(AppError):
        decode_cursor("eyJhIjogMX0=")  # Valid base64, missing keys
    with pytest.raises(AppError):
        decode_cursor("eyJ0IjogImJhZHRpbWUiLCAiaWQiOiAiYSJ9")  # Bad time string
