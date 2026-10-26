from hashlib import sha256


def stable_id(prefix: str, idempotency_key: str) -> str:
    digest = sha256(idempotency_key.encode("utf-8")).hexdigest()
    return f"{prefix}-{digest}"
