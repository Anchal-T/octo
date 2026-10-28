from hashlib import sha256


def stable_id(prefix: str, idempotency_key: str) -> str:
    digest = sha256(idempotency_key.encode("utf-8")).hexdigest()
    return f"{prefix}-{digest}"


def closure_id(fact_id: str, closing_commit: str) -> str:
    digest = sha256(f"{fact_id}|{closing_commit}".encode("utf-8")).hexdigest()
    return f"fc-{digest[:20]}"
