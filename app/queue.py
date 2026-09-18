import asyncio
import logging
import os
from dataclasses import dataclass

from .diff import find_comment_line, parse_files
from .github import GithubClient
from .reviewer import review_diff


logger = logging.getLogger(__name__)

# TEMPORARY: allow redelivered webhooks for superseded commits to be
# reviewed instead of discarded as stale. Set ALLOW_STALE_REVIEWS=1.
# Note the file list still reflects the PR's current head while
# sources are read at the queued commit, so line anchoring can be
# slightly off — fine for replaying a missed review, not for normal
# operation.
ALLOW_STALE_REVIEWS = os.getenv("ALLOW_STALE_REVIEWS", "0") == "1"

github = GithubClient()


@dataclass
class ReviewJob:
    owner: str
    repo: str
    pr_number: int
    installation_id: int
    commit_sha: str


review_queue: asyncio.Queue[ReviewJob] = asyncio.Queue()

# Keys (owner, repo, pr_number, commit_sha) currently queued or being
# processed. Same event loop, so plain set ops are race-free. Prevents
# GitHub's duplicate deliveries (retries, sync+reopened races) from
# stacking identical reviews behind a slow local model.
pending_jobs: set[tuple[str, str, int, str]] = set()


def job_key(
    owner: str, repo: str, pr_number: int, commit_sha: str
) -> tuple[str, str, int, str]:
    return (owner, repo, pr_number, commit_sha)


async def enqueue_review(job: ReviewJob) -> bool:
    """Queue a review. Returns False if the identical job is already
    pending — the caller should skip (no duplicate confirm comment)."""
    key = job_key(job.owner, job.repo, job.pr_number, job.commit_sha)

    if key in pending_jobs:
        print(f"Skipping duplicate review: {key} already pending")
        return False

    pending_jobs.add(key)
    await review_queue.put(job)

    print(
        f"Queued review: {job.owner}/{job.repo}#{job.pr_number} "
        f"(queue depth: {review_queue.qsize()})"
    )

    return True


async def worker():
    logger.info("Review worker started")

    while True:
        job = await review_queue.get()

        try:
            await process_review(job)

        except Exception as exc:
            logger.exception(
                "Review failed: %s/%s#%s",
                job.owner,
                job.repo,
                job.pr_number,
            )
            print(
                f"Review failed: {job.owner}/{job.repo}"
                f"#{job.pr_number}: {exc}"
            )

        finally:
            review_queue.task_done()

async def process_review(job: ReviewJob):
    """Run one review; always releases the pending key (all exits)."""
    try:
        await _process_review_inner(job)
    finally:
        pending_jobs.discard(
            job_key(
                job.owner,
                job.repo,
                job.pr_number,
                job.commit_sha,
            )
        )

async def _process_review_inner(job: ReviewJob):
    print(
        f"Starting review: {job.owner}/{job.repo}"
        f"#{job.pr_number} @ {job.commit_sha[:8]}"
    )
    logger.info(
        "Starting review: %s/%s#%s @ %s",
        job.owner,
        job.repo,
        job.pr_number,
        job.commit_sha[:8],
    )

    token = await github.create_installation_token(
        job.installation_id
    )

    # -------------------------------------------------
    # Check 1: Has the PR changed while this job waited?
    # -------------------------------------------------

    current_sha = await github.get_pr_head_sha(
        job.owner,
        job.repo,
        job.pr_number,
        token,
    )

    if current_sha != job.commit_sha:
        if ALLOW_STALE_REVIEWS:
            print(
                f"Allowing stale review (ALLOW_STALE_REVIEWS=1): "
                f"{job.owner}/{job.repo}#{job.pr_number} "
                f"(queued {job.commit_sha[:8]}, "
                f"head is now {current_sha[:8]})"
            )
        else:
            logger.info(
                "Discarding stale review: %s/%s#%s",
                job.owner,
                job.repo,
                job.pr_number,
            )
            print(
                f"Discarding stale review: {job.owner}/{job.repo}"
                f"#{job.pr_number} (queued {job.commit_sha[:8]}, "
                f"head is now {current_sha[:8]})"
            )
            return

    # -------------------------------------------------
    # Fetch PR files
    # -------------------------------------------------

    files = await github.get_pr_files(
        job.owner,
        job.repo,
        job.pr_number,
        token,
    )

    if not files:
        print(
            f"No changed files for {job.owner}/{job.repo}"
            f"#{job.pr_number}, posting empty review"
        )
        await github.post_review(
            job.owner,
            job.repo,
            job.pr_number,
            token,
            job.commit_sha,
            [],
        )
        return

    # -------------------------------------------------
    # Ask the model
    # -------------------------------------------------

    review = await review_diff(
        files,
        github,
        job.owner,
        job.repo,
        token,
        job.commit_sha,
    )

    # -------------------------------------------------
    # Check 2: Did a new commit arrive while
    # the model was thinking?
    # -------------------------------------------------

    current_sha = await github.get_pr_head_sha(
        job.owner,
        job.repo,
        job.pr_number,
        token,
    )

    if current_sha != job.commit_sha and not ALLOW_STALE_REVIEWS:
        logger.info(
            "Discarding review because PR changed "
            "during model execution: %s/%s#%s",
            job.owner,
            job.repo,
            job.pr_number,
        )
        print(
            f"Discarding review: {job.owner}/{job.repo}"
            f"#{job.pr_number} changed during model execution "
            f"(now {current_sha[:8]})"
        )
        return

    # -------------------------------------------------
    # Validate every finding against the actual diff
    # -------------------------------------------------

    parsed_files = parse_files(files)

    valid_findings = []

    for finding in review.findings:
        matched = find_comment_line(
            parsed_files,
            finding.path,
            finding.line,
            finding.side,
        )

        if matched:
            valid_findings.append(finding)
        else:
            logger.warning(
                "Discarding invalid finding: %s:%s (%s)",
                finding.path,
                finding.line,
                finding.side,
            )

    # -------------------------------------------------
    # Post GitHub review
    # -------------------------------------------------

    await github.post_review(
        job.owner,
        job.repo,
        job.pr_number,
        token,
        job.commit_sha,
        valid_findings,
    )

    logger.info(
        "Review complete: %s/%s#%s — %d findings",
        job.owner,
        job.repo,
        job.pr_number,
        len(valid_findings),
    )
    print(
        f"Review complete: {job.owner}/{job.repo}"
        f"#{job.pr_number} — {len(valid_findings)} finding(s)"
    )