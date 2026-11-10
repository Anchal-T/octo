from hashlib import sha256


def stable_id(prefix: str, idempotency_key: str) -> str:
    digest = sha256(idempotency_key.encode()).hexdigest()
    return f"{prefix}-{digest}"


def workflow_key(issue_id: str, attempt: int) -> str:
    if not 1 <= attempt <= 99:
        raise ValueError("attempt must be between 1 and 99")
    if attempt == 1:
        return issue_id
    return f"{issue_id}-r{attempt}"


def closure_id(fact_id: str, closing_commit: str) -> str:
    digest = sha256(f"{fact_id}|{closing_commit}".encode()).hexdigest()
    return f"fc-{digest[:20]}"
