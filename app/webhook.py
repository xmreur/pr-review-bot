import hashlib, hmac

from fastapi import HTTPException

def verify_github_signature(body: bytes, signature: str | None, secret: str) -> bool:
    if not signature:
        raise HTTPException(status_code=403, detail="Missing webhook signature")

    expected = (
        "sha256="
        + hmac.new(
            secret.encode("utf-8"),
            body,
            hashlib.sha256,
        ).hexdigest()
    )

    if not hmac.compare_digest(expected, signature):
        raise HTTPException(
            status_code=403,
            detail="Invalid webhook signature",
        )