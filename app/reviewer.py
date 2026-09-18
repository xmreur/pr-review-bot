"""PR reviewer optimized for a small (3B) model in a 16k-token window.

Design goals:
  1. Never silently overflow the context window (cost model counts
     diff + context, not diff alone).
  2. Keep batches small: a 3B model reasons better over 3k tokens
     than over 12k tokens. Filling the window hurts accuracy.
  3. Keep related hunks together (group by file).
  4. Give compact, high-signal context (merged ranges, exact changed
     markers, enclosing scope header) instead of blind +/-30 lines.
  5. Two passes: propose candidates, then verify each one with a
     focused prompt. The verify pass is the main false-positive
     killer and costs little because there are few candidates.
"""

import asyncio
import json
import logging
import re

from .diff import (
    build_numbered_diff,
    extract_focused_context,
    is_skippable_file,
    is_test_file,
    normalize_code,
    parse_patch,
    split_patch_into_hunks,
)
from .llm import ask_llm
from .models import Finding, Review


logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Budgets. 16k tokens ~= 60-64k chars for code (~4 chars/token).
# We deliberately use far less per call: small models degrade with
# long inputs ("lost in the middle"), so several small precise calls
# beat one giant call.
# ------------------------------------------------------------------

# Diff text allowed per review call.
MAX_DIFF_CHARS = 6000
# Context text allowed per review call (shared by all files in it).
MAX_CONTEXT_CHARS = 8000
# Hard ceiling for diff + context in one prompt.
MAX_TOTAL_CHARS = 15000

# Context shaping.
BASE_PAD = 12  # lines on each side of a changed line cluster
MAX_CONTEXT_CHARS_PER_FILE = 6000
MAX_SINGLE_HUNK_CHARS = 12000  # oversized hunk -> truncate, don't blow batch

VERIFY_MAX_CHARS = 6000  # focused verify prompt per batch

# Test/spec files are skipped: commenting on mocks and assertions at
# src severity is pure noise. Set True to review them anyway.
REVIEW_TEST_FILES = False

# Speed: concurrency for the two I/O-bound stages. File fetches
# (GitHub API) parallelize well. LLM batches do NOT: the local
# llama.cpp server runs inference serially, so concurrent review
# calls just pile up in its queue while each caller's timeout ticks
# — that is what caused the ReadTimeout storm. Keep inference
# serial; the speedup comes from fewer calls (test skip, tight
# max_tokens) and parallel fetching.
FETCH_CONCURRENCY = 8
REVIEW_CONCURRENCY = 1

SYSTEM_PROMPT = """You are a senior engineer reviewing a pull request diff.
Report ONLY real bugs introduced by THIS diff.

Focus: crashes, wrong behavior, security holes, data loss, race
conditions, resource leaks, broken error handling, broken API contracts.

NEVER report: style, formatting, naming, refactoring suggestions,
missing tests/docs, or anything that already existed before the diff.
NEVER report a problem whose fix belongs in a different function.
NEVER report race conditions / thread-safety for sequential awaited
calls (e.g. CLI prompts like select/text/confirm). JS/TS is
single-threaded; sequential awaits cannot race.
NEVER report cosmetic issues: empty logging output, confusing but
harmless messages, or empty-array/empty-string handling that does not
throw, corrupt data, or break a contract.

Rules:
- Comment ONLY on lines starting with '+' in CHANGED DIFF.
- side MUST be "RIGHT". line MUST be the exact number shown.
- Quote the exact changed code in "evidence".
- Give "confidence": high, medium, or low.
- If unsure, return NO finding. Empty findings is the correct answer
  when nothing is clearly wrong.
- At most 5 findings. Highest severity first.

Return ONLY JSON: {"findings": [{"path": str, "line": int,
"side": "RIGHT", "severity": "low|medium|high|critical",
"confidence": "low|medium|high", "title": str,
"body": str, "evidence": str}]}
If clean: {"findings": []}"""

VERIFY_PROMPT = """You verify code-review findings. Drop anything uncertain.
For each candidate, check the claim against the code and answer
"keep" ONLY if ALL are true:
1. The quoted evidence appears in a '+' line of CHANGED DIFF.
2. There is a concrete execution path to observable wrong behavior.
3. Surrounding code does NOT already prevent it.
Otherwise answer "drop".

Return ONLY JSON: {"verdicts": [{"path": str, "line": int,
"verdict": "keep|drop", "reason": str}]}"""


def _changed_lines_of_patch(path: str, patch: str) -> list[int]:
    parsed = parse_patch(path, patch)
    return [
        line.line
        for line in parsed.lines
        if line.changed and line.side == "RIGHT"
    ]


def _estimate_context_chars(
    source: str | None,
    changed: list[int],
) -> int:
    """Cheap pre-fetch estimate so batching accounts for context."""
    if not source or not changed:
        return 0
    span = max(changed) - min(changed) + 1
    # Merged-range heuristic: span + padding, ~90 chars/line.
    lines = min(span + 2 * BASE_PAD, len(source.splitlines()))
    return min(lines * 90, MAX_CONTEXT_CHARS_PER_FILE)


def build_review_batches(
    files: list[dict],
    max_diff_chars: int = MAX_DIFF_CHARS,
    max_context_chars: int = MAX_CONTEXT_CHARS,
) -> list[list[dict]]:
    """Group hunks into small, file-coherent batches.

    - Skips unreviewable files (lockfiles, generated, huge).
    - Never splits a file across batches unless it alone exceeds
      the budget (then splits by hunk).
    - Accounts for estimated context cost, not just diff size, so
      the final prompt cannot silently overflow 16k.
    """
    # Group hunks per file first (preserves related changes).
    per_file: list[tuple[dict, list[dict], int, int]] = []

    for file in files:
        patch = file.get("patch")
        if not patch:
            continue

        filename = file["filename"]

        if is_skippable_file(filename, patch):
            continue

        if not REVIEW_TEST_FILES and is_test_file(filename):
            continue

        hunks = split_patch_into_hunks(filename, patch)
        if not hunks:
            continue

        source = file.get("source")
        file_diff_chars = 0
        file_ctx_chars = 0

        for hunk in hunks:
            hunk["source"] = source
            hunk_diff = build_numbered_diff([hunk])
            if len(hunk_diff) > MAX_SINGLE_HUNK_CHARS:
                # Truncate pathological hunks (e.g. generated files
                # misdetected) instead of letting one hunk eat the
                # whole window.
                hunk["patch"] = "\n".join(
                    hunk["patch"].splitlines()[:200]
                )
                hunk_diff = build_numbered_diff([hunk])
            changed = _changed_lines_of_patch(
                filename, hunk["patch"]
            )
            hunk["_changed"] = changed
            hunk["_diff_chars"] = len(hunk_diff)
            hunk["_ctx_chars"] = _estimate_context_chars(
                source, changed
            )
            file_diff_chars += len(hunk_diff)
            file_ctx_chars += hunk["_ctx_chars"]

        per_file.append(
            (file, hunks, file_diff_chars, file_ctx_chars)
        )

    # Largest files first packs more evenly (first-fit decreasing).
    per_file.sort(
        key=lambda t: t[2] + t[3], reverse=True
    )

    batches: list[list[dict]] = []
    batch_diff = 0
    batch_ctx = 0
    current: list[dict] = []

    def flush():
        nonlocal current, batch_diff, batch_ctx
        if current:
            batches.append(current)
            current = []
            batch_diff = 0
            batch_ctx = 0

    for _file, hunks, f_diff, f_ctx in per_file:
        # Whole file fits in current batch -> keep together.
        if (
            current
            and (
                batch_diff + f_diff > max_diff_chars
                or batch_ctx + f_ctx > max_context_chars
                or batch_diff + f_diff + batch_ctx + f_ctx
                > MAX_TOTAL_CHARS
            )
        ):
            flush()

        if (
            f_diff <= max_diff_chars
            and f_ctx <= max_context_chars
        ):
            # Fits as a unit (maybe after flush above).
            if (
                batch_diff + f_diff > max_diff_chars
                or batch_ctx + f_ctx > max_context_chars
            ):
                flush()
            current.extend(hunks)
            batch_diff += f_diff
            batch_ctx += f_ctx
        else:
            # Single file exceeds budget: split by hunk.
            for hunk in hunks:
                d = hunk["_diff_chars"]
                c = hunk["_ctx_chars"]
                if current and (
                    batch_diff + d > max_diff_chars
                    or batch_ctx + c > max_context_chars
                ):
                    flush()
                current.append(hunk)
                batch_diff += d
                batch_ctx += c

    flush()

    # Strip bookkeeping keys.
    for batch in batches:
        for hunk in batch:
            hunk.pop("_changed", None)
            hunk.pop("_diff_chars", None)
            hunk.pop("_ctx_chars", None)

    return batches


def build_context_block(
    path: str,
    patch: str,
    source: str,
) -> str:
    """Compact context for one hunk's file (kept for compatibility)."""
    changed = _changed_lines_of_patch(path, patch)
    if not changed or not source:
        return ""

    context = extract_focused_context(
        source,
        changed,
        pad=BASE_PAD,
        max_chars=MAX_CONTEXT_CHARS_PER_FILE,
    )
    if not context:
        return ""

    return f"FILE: {path}\n{context}"


def _merge_batch_context(batch: list[dict]) -> str:
    """Merge context per file so overlapping hunks share one block."""
    by_file: dict[str, tuple[str, set[int]]] = {}

    for hunk in batch:
        source = hunk.get("source")
        if not source:
            continue
        path = hunk["filename"]
        changed = _changed_lines_of_patch(path, hunk["patch"])
        if not changed:
            continue
        if path not in by_file:
            by_file[path] = (source, set())
        by_file[path][1].update(changed)

    blocks = []
    used = 0

    for path, (source, changed_set) in by_file.items():
        # Share the per-batch context budget across files.
        remaining = MAX_CONTEXT_CHARS - used
        if remaining < 500:
            break
        context = extract_focused_context(
            source,
            sorted(changed_set),
            pad=BASE_PAD,
            max_chars=min(
                MAX_CONTEXT_CHARS_PER_FILE, remaining
            ),
        )
        if not context:
            continue
        block = f"FILE: {path}\n{context}"
        blocks.append(block)
        used += len(block)

    return "\n\n".join(blocks)


def _parse_json_response(response: str) -> dict:
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass
    # Small models often wrap JSON in fences despite instructions.
    m = re.search(r"\{.*\}", response, re.DOTALL)
    if m:
        return json.loads(m.group(0))
    print("LLM returned invalid JSON:")
    print(response[:2000])
    raise ValueError("LLM returned invalid JSON")


async def review_batch(batch: list[dict]) -> Review:
    numbered_diff = build_numbered_diff(batch)
    context_text = _merge_batch_context(batch)

    # Safety: hard-truncate rather than overflow the window.
    if len(numbered_diff) > MAX_DIFF_CHARS + 2000:
        numbered_diff = (
            numbered_diff[: MAX_DIFF_CHARS + 2000]
            + "\n...[diff truncated]"
        )
    if len(context_text) > MAX_CONTEXT_CHARS + 2000:
        context_text = (
            context_text[: MAX_CONTEXT_CHARS + 2000]
            + "\n...[context truncated]"
        )

    user_prompt = (
        "Review this pull request portion.\n"
        "Use SURROUNDING SOURCE only to check control flow and "
        "defaults. Report ONLY '+' lines from CHANGED DIFF.\n\n"
        f"CHANGED DIFF:\n{numbered_diff}\n\n"
        f"SURROUNDING SOURCE (>>> marks changed lines):\n"
        f"{context_text or '(no file content available)'}"
    )

    response = await ask_llm(
        SYSTEM_PROMPT, user_prompt, max_tokens=1000, temperature=0.0
    )
    data = _parse_json_response(response)
    return Review.model_validate(data)


async def verify_batch(
    batch: list[dict],
    candidates: list[Finding],
) -> list[Finding]:
    """Second pass: re-check each candidate with focused evidence.

    Returns only candidates with verdict 'keep'. Falls back to the
    unfiltered list if the verifier output is unusable (fail-open
    would spam; fail-closed to low-confidence filter instead).
    """
    if not candidates:
        return []

    numbered_diff = build_numbered_diff(batch)
    context_text = _merge_batch_context(batch)

    claim_lines = []
    for f in candidates:
        claim_lines.append(
            f"- {f.path}:{f.line} [{f.severity}] {f.title}\n"
            f"  Evidence: {(f.evidence or f.body)[:600]}"
        )
    claims = "\n".join(claim_lines)

    prompt = (
        "CHANGED DIFF:\n"
        f"{numbered_diff[:MAX_DIFF_CHARS]}\n\n"
        "SURROUNDING SOURCE:\n"
        f"{context_text[:MAX_CONTEXT_CHARS]}\n\n"
        f"CANDIDATE FINDINGS:\n{claims}"
    )
    # Keep the verify prompt small: it must fit alongside diff.
    prompt = prompt[:VERIFY_MAX_CHARS + MAX_DIFF_CHARS]

    try:
        response = await ask_llm(
            VERIFY_PROMPT, prompt, max_tokens=400, temperature=0.0
        )
        data = _parse_json_response(response)
    except Exception as exc:
        print(f"Verify pass failed, keeping filtered set: {exc}")
        return [f for f in candidates if f.confidence != "low"]

    verdicts = {
        (v.get("path"), v.get("line")): v.get("verdict")
        for v in data.get("verdicts", [])
        if isinstance(v, dict)
    }

    # Fail-closed: only an explicit "keep" survives. A missing
    # verdict means the verifier could not confirm the claim, and
    # unconfirmed claims are exactly the false positives we drop.
    return [
        f
        for f in candidates
        if verdicts.get((f.path, f.line)) == "keep"
        and f.confidence != "low"
    ]


def _strict_filter(
    batch: list[dict], findings: list[Finding]
) -> list[Finding]:
    """Drop findings that cannot be posted or are low-signal."""
    valid: set[tuple[str, int]] = set()
    for hunk in batch:
        parsed = parse_patch(hunk["filename"], hunk["patch"])
        for line in parsed.lines:
            if line.changed and line.side == "RIGHT":
                valid.add((line.path, line.line))

    out = []
    for f in findings:
        if f.side != "RIGHT":
            continue
        if (f.path, f.line) not in valid:
            continue
        if (f.confidence or "medium") == "low":
            continue
        out.append(f)

    return out


def _evidence_anchor(evidence: str | None) -> str:
    """Pick a matchable code fragment from the quoted evidence.

    Uses the longest substantive line so a multi-line quote still
    anchors to one real diff line.
    """
    if not evidence:
        return ""
    best = ""
    for raw in evidence.splitlines():
        line = raw.strip().lstrip("+->| ").strip("`'\"")
        norm = normalize_code(line)
        if len(norm) > len(best):
            best = norm
    return best if len(best) >= 10 else ""


def _ground_findings(
    batch: list[dict], findings: list[Finding]
) -> list[Finding]:
    """Fix the 'wrong code snippet' class of false positives.

    - Drops findings whose evidence matches no '+' line in the file.
    - Snaps the comment line to the '+' line the evidence actually
      came from, so the GitHub annotation lands on the right code.
    """
    plus_lines: dict[str, dict[int, str]] = {}
    for hunk in batch:
        parsed = parse_patch(hunk["filename"], hunk["patch"])
        bucket = plus_lines.setdefault(hunk["filename"], {})
        for line in parsed.lines:
            if line.changed and line.side == "RIGHT":
                bucket[line.line] = normalize_code(line.content)

    out = []
    for f in findings:
        bucket = plus_lines.get(f.path, {})
        if not bucket:
            continue
        anchor = _evidence_anchor(f.evidence)
        if not anchor:
            out.append(f)
            continue
        attached = bucket.get(f.line, "")
        if anchor in attached or attached in anchor:
            out.append(f)
            continue
        # Wrong line: find the '+' line the evidence came from.
        fixed = next(
            (
                ln
                for ln, content in bucket.items()
                if anchor in content or content in anchor
            ),
            None,
        )
        if fixed is None:
            logger.warning(
                "Discarding ungrounded finding: %s:%s (%s)",
                f.path,
                f.line,
                f.title[:60],
            )
            continue
        out.append(f.model_copy(update={"line": fixed}))

    return out


# Claims that are almost always 3B hallucinations unless the diff
# contains matching primitives. Checked deterministically because
# the model will not police itself.
_CONCURRENCY_TOKENS = (
    "promise.all",
    "promise.race",
    "worker",
    "sharedarraybuffer",
    "atomics",
    "settimeout",
    "setinterval",
    "mutex",
    "lock(",
    "thread",
    "spawn",
    "fork(",
    "cluster",
)

_RACE_RE = re.compile(
    r"race condition|thread-safe|threadsafe|concurrent|concurrency"
    r"|data race|deadlock",
    re.IGNORECASE,
)

_COSMETIC_RE = re.compile(
    r"log\s+(an?\s+)?empty|empty\s+(string|array).*log|confus",
    re.IGNORECASE,
)

_HEDGE_RE = re.compile(
    r"\bcould\b|\bmight\b|\bmay\b|potentially|possibly|unclear"
    r"|unexpected behavior",
    re.IGNORECASE,
)

_CONCRETE_RE = re.compile(
    r"throw|exception|typeerror|referenceerror|rangeerror|crash"
    r"|exploit|xss|injection|leak|deadlock|data loss|corrupt"
    r"|cannot read|undefined is not|null is not|denial of service"
    r"|privilege|auth bypass",
    re.IGNORECASE,
)


def _drop_speculative(
    batch: list[dict], findings: list[Finding]
) -> list[Finding]:
    """Drop hallucination-shaped claims without touching the model."""
    diff_text = normalize_code(build_numbered_diff(batch)).lower()

    out = []
    for f in findings:
        text = f"{f.title}\n{f.body}"
        # Title words are cheap ("Uncaught exception" on a logging
        # complaint); only the body can vouch for concreteness.
        body_concrete = bool(_CONCRETE_RE.search(f.body or ""))
        if _RACE_RE.search(text) and not any(
            tok in diff_text for tok in _CONCURRENCY_TOKENS
        ):
            logger.warning(
                "Discarding speculative race claim: %s:%s",
                f.path,
                f.line,
            )
            continue
        if _COSMETIC_RE.search(text) and not body_concrete:
            logger.warning(
                "Discarding cosmetic claim: %s:%s",
                f.path,
                f.line,
            )
            continue
        if (
            f.severity in ("high", "critical")
            and _HEDGE_RE.search(text)
            and not body_concrete
        ):
            logger.warning(
                "Discarding hedged high-severity claim: %s:%s",
                f.path,
                f.line,
            )
            continue
        out.append(f)

    return out


async def review_diff(
    files: list[dict],
    github,
    owner: str,
    repo: str,
    token: str,
    commit_sha: str,
) -> Review:

    async def _fetch(file: dict, sem: asyncio.Semaphore) -> dict | None:
        patch = file.get("patch")

        if not patch:
            return None

        if is_skippable_file(file["filename"], patch):
            return None

        if not REVIEW_TEST_FILES and is_test_file(file["filename"]):
            logger.info("Skipping test file: %s", file["filename"])
            return None

        async with sem:
            try:
                source = await github.get_file_content(
                    owner,
                    repo,
                    file["filename"],
                    token,
                    commit_sha,
                )
            except Exception as exc:
                print(
                    f"Could not fetch source for "
                    f"{file['filename']}: {exc}"
                )
                source = None

        return {**file, "source": source}

    fetch_sem = asyncio.Semaphore(FETCH_CONCURRENCY)
    fetched = await asyncio.gather(
        *(_fetch(file, fetch_sem) for file in files)
    )
    enriched_files = [f for f in fetched if f is not None]

    batches = build_review_batches(enriched_files)

    print(f"PR split into {len(batches)} review batch(es)")

    review_sem = asyncio.Semaphore(REVIEW_CONCURRENCY)

    async def _process_batch(
        index: int, batch: list[dict]
    ) -> list[Finding]:
        # One slow/dead batch must not kill the whole review.
        try:
            async with review_sem:
                print(f"Reviewing batch {index} ({len(batch)} hunk(s))")
                review = await review_batch(batch)
        except Exception as exc:
            logger.warning("Batch %d review failed: %s", index, exc)
            return []

        candidates = _strict_filter(batch, review.findings)
        grounded = _ground_findings(batch, candidates)
        plausible = _drop_speculative(batch, grounded)
        try:
            verified = await verify_batch(batch, plausible)
        except Exception as exc:
            logger.warning("Batch %d verify failed: %s", index, exc)
            verified = plausible

        print(
            f"Batch {index}: {len(review.findings)} candidate(s), "
            f"{len(verified)} kept"
        )
        return verified

    # Concurrent batch reviews; a batch with zero candidates makes
    # zero verifier calls.
    results = await asyncio.gather(
        *(
            _process_batch(index, batch)
            for index, batch in enumerate(batches, start=1)
        )
    )

    seen: set[tuple[str, int, str]] = set()
    all_findings: list[Finding] = []
    for verified in results:
        for f in verified:
            key = (f.path, f.line, f.title.strip().lower())
            if key in seen:
                continue
            seen.add(key)
            all_findings.append(f)

    # Cap total comments: GitHub reviews get noisy fast.
    all_findings.sort(
        key=lambda f: (
            {"critical": 0, "high": 1, "medium": 2, "low": 3}[
                f.severity
            ],
            f.path,
            f.line,
        )
    )
    return Review(findings=all_findings[:10])
