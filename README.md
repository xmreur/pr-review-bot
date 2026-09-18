# PR Reviewer Bot

A GitHub App that posts AI code reviews on pull requests, powered by a **local** LLM (Qwen 2.5 Coder 3B via `llama.cpp`, 16k context). No code ever leaves your machine.

## How it works

1. GitHub sends a `pull_request` webhook (`opened`, `synchronize`, `reopened`).
2. The app fetches the PR diff, posts an "AI review started" comment, and queues a review job.
3. A serial background worker splits the diff into small file-coherent batches, reviews each with the local model, filters the findings, and posts one GitHub review with inline comments.

Each batch goes through a defense-in-depth pipeline designed for a small model:

- **Propose** — small prompt (~3.5k tokens, not a full 16k window; small models reason better small), compact merged context with exact `>>>` changed-line markers plus enclosing function scope.
- **Strict filter** — finding must sit on a real `+` line, `RIGHT` side, non-`low` confidence.
- **Grounding** — quoted evidence must fuzzy-match a `+` line; the comment line is snapped to the line the evidence came from, ungrounded claims are dropped.
- **Speculative filter** — deterministic bans on hallucination-shaped claims (race conditions in sequential awaits, cosmetic logging complaints, hedged high-severity claims).
- **Verify pass** — fail-closed second LLM opinion; only explicit `keep` survives.
- Test/spec files are skipped by default; max 10 findings per PR, highest severity first.

Stale jobs (new commit arrived while queued or mid-review) are discarded; duplicate deliveries for the same commit are deduplicated.

## Requirements

- Python 3.12+ with the venv in `.venv` (`fastapi`, `httpx`, `pyjwt`, `pydantic`, `python-dotenv`)
- A `llama.cpp` server on `127.0.0.1:8080` serving a Qwen-compatible chat model (`qwen2.5-coder-3b-instruct-q4_k_m.gguf` works; 16k context)
- A GitHub App with **Pull requests: Read & write**, subscribed to **Pull request** events, installed on the target repos

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install fastapi "uvicorn[standard]" httpx pyjwt pydantic python-dotenv

# llama.cpp server (from the llama.cpp directory)
./llama-server -m ../models/qwen2.5-coder-3b-instruct-q4_k_m.gguf \
  --ctx-size 16384 -c 16384 --port 8080
```

Create `.env` (never committed — see `.gitignore`):

```env
GITHUB_APP_ID="123456"
GITHUB_PRIVATE_KEY_PATH="pr-reviewer-pk.pem"
GITHUB_WEBHOOK_SECRET="your-webhook-secret"
HTTP_PORT=57344
```

Download the app's private key from the GitHub App settings into `GITHUB_PRIVATE_KEY_PATH`, set the webhook URL to `http(s)://<host>:<port>/webhook` with the same secret, then:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 57344
```

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `GITHUB_APP_ID` | — | GitHub App ID |
| `GITHUB_PRIVATE_KEY_PATH` | — | Path to the App private key (`.pem`) |
| `GITHUB_WEBHOOK_SECRET` | — | Webhook HMAC secret |
| `HTTP_PORT` | — | Port the server listens on |
| `ALLOW_STALE_REVIEWS` | `0` | `1` = review superseded commits (for replaying redelivered webhooks); off by default |

Tuning knobs live in `app/reviewer.py` (`MAX_DIFF_CHARS`, `MAX_CONTEXT_CHARS`, `BASE_PAD`, `REVIEW_TEST_FILES`) and `app/queue.py` (`FETCH_CONCURRENCY`, `REVIEW_CONCURRENCY` — keep review at 1 against a serial `llama.cpp` server).

## Endpoints

- `GET /` — liveness
- `GET /health` — worker state: `worker_alive`, `queue_depth`, `pending` jobs
- `POST /webhook` — GitHub event receiver (HMAC-verified)

## Repo layout

```
app/
  main.py      webhook receiver + app wiring
  queue.py     serial job queue, staleness/dedupe guards
  reviewer.py  batching, prompts, propose→filter→verify pipeline
  diff.py      patch parsing, smart context extraction, skip lists
  github.py    App auth, PR files, review posting
  llm.py       llama.cpp chat client (retries, long timeout)
  models.py    Finding / Review schemas
test_gh.py         end-to-end drill against a real PR
test_reviewer.py   reviewer smoke test
```

## License

MIT — see [LICENSE](LICENSE).
