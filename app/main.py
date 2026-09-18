import asyncio
from contextlib import asynccontextmanager
import os

from fastapi import FastAPI, Request

from app.webhook import verify_github_signature
from .github import GithubClient
from .diff import build_diff
from .reviewer import review_diff

from .queue import ReviewJob, enqueue_review, job_key, pending_jobs, review_queue, worker

from dotenv import load_dotenv

load_dotenv()

GITHUB_WEBHOOK_SECRET = os.getenv("GITHUB_WEBHOOK_SECRET")

@asynccontextmanager
async def lifespan(app: FastAPI):
    worker_task = asyncio.create_task(worker())
    app.state.worker_task = worker_task

    print("Review worker started")

    yield

    worker_task.cancel()

    try:
        await worker_task 
    except asyncio.CancelledError:
        pass

app = FastAPI(lifespan=lifespan)

github = GithubClient()


@app.get("/")
async def root():
    return {"status": "ok"}


@app.get("/health")
async def health():
    """Inspect worker state: a queued-but-never-starting review shows
    up here as growing queue_depth / stale pending entries."""
    return {
        "status": "ok",
        "worker_alive": not worker_task.done() if (worker_task := getattr(app.state, "worker_task", None)) else False,
        "queue_depth": review_queue.qsize(),
        "pending": [f"{o}/{r}#{n}@{s[:8]}" for o, r, n, s in pending_jobs],
    }


@app.post("/webhook")
async def webhook(request: Request):
    # ---------------------------------------------
    # 1. Read the exact raw request body
    # ---------------------------------------------

    body = await request.body()

    # ---------------------------------------------
    # 2. Verify GitHub signature
    # ---------------------------------------------

    signature = request.headers.get(
        "X-Hub-Signature-256"
    )

    verify_github_signature(
        body,
        signature,
        GITHUB_WEBHOOK_SECRET,
    )

    # ---------------------------------------------
    # 3. Only parse JSON AFTER verification
    # ---------------------------------------------
    payload = await request.json()

    event = request.headers.get("X-GitHub-Event")

    if event != "pull_request":
        return {"ok": True}

    action = payload["action"]

    if action not in {
        "opened",
        "synchronize",
        "reopened",
    }:
        return {"ok": True}

    pull_request = payload["pull_request"]
    repository = payload["repository"]

    owner = repository["owner"]["login"]
    repo = repository["name"]

    pr_number = pull_request["number"]
    commit_sha = pull_request["head"]["sha"]

    print(
        f"Review requested: {owner}/{repo} PR #{pr_number}"
    )

    # Duplicate delivery (retry, redelivery) for a commit already
    # queued or under review: skip before any API calls or comments.
    if job_key(owner, repo, pr_number, commit_sha) in pending_jobs:
        print(f"Duplicate delivery for {commit_sha[:8]}, skipping")
        return {"ok": True, "queued": False, "duplicate": True}

    installation_id = await github.get_installation_id(owner, repo)

    

    token = await github.create_installation_token(installation_id)

    files = await github.get_pr_files(owner, repo, pr_number, token)

    print(f"Found {len(files)} changed files")

    diff = build_diff(files)

    print("\n======= DIFF =======")
    print(diff)
    print("==================\n")

    await github.post_confirm(owner, repo, pr_number, token)

    # TODO: Send diff to AI model
    queued = await enqueue_review(
        ReviewJob(
            owner=owner,
            repo=repo,
            pr_number=pr_number,
            installation_id=installation_id,
            commit_sha=commit_sha,
        )
    )

    return {"ok": True, "queued": queued }

