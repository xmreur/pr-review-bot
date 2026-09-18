from re import A
import os, time, httpx, jwt

from dotenv import load_dotenv

load_dotenv()

GITHUB_API = "https://api.github.com"

APP_ID = os.environ['GITHUB_APP_ID']
PRIVATE_KEY_PATH = os.environ['GITHUB_PRIVATE_KEY_PATH']

class GithubClient:

    def __init__(self):
        with open(PRIVATE_KEY_PATH, 'rb') as f:
            self.private_key = f.read()

    def create_app_jwt(self) -> str:
        now = int(time.time())

        payload = {
            'iat': now - 60,
            'exp': now + 600,
            'iss': APP_ID
        }

        return jwt.encode(payload, self.private_key, algorithm='RS256')

    async def get_installation_id(self, owner: str, repo: str) -> int:
        jwt_token = self.create_app_jwt()

        headers = {
            "Authorization": f"Bearer {jwt_token}",
            "Accept": "application/vnd.github+json",
        }

        url = (
            f"{GITHUB_API}/repos/"
            f"{owner}/{repo}/installation"
        )

        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers)

            response.raise_for_status()

            data = response.json()

            return data['id']

    async def create_installation_token(self, installation_id: int) -> str:
        jwt_token = self.create_app_jwt()

        headers = {
            "Authorization": f"Bearer {jwt_token}",
            "Accept": "application/vnd.github+json",
        }

        url = (
            f"{GITHUB_API}/app/installations/"
            f"{installation_id}/access_tokens"
        )
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers)

            response.raise_for_status()

            data = response.json()

            return data['token']

    async def get_pr_files(self, owner: str, repo: str, pr_number: int, token: str) -> list[str]:
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
        }

        url = (
            f"{GITHUB_API}/repos/"
            f"{owner}/{repo}/pulls/"
            f"{pr_number}/files"
        )

        files = []

        async with httpx.AsyncClient() as client:
 
            page = 1

            while True:
                response = await client.get(
                    url,
                    headers=headers,
                    params = {
                        'per_page': 100,
                        "page": page
                    }
                )

                response.raise_for_status()

                batch = response.json()

                if not batch: break

                files.extend(batch)

                if len(batch) < 100:
                    break

                page += 1

        return files

    async def post_confirm(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        token: str,
    ) -> dict:
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-Github-Api-Version": "2026-03-10",
        }

        url = (
            f"{GITHUB_API}/repos/"
            f"{owner}/{repo}/issues/"
            f"{pr_number}/comments"
        )

        payload = {
            "body": (
                "🤖 **AI review started.**\n\n"
                "I'm analyzing the changes in this pull request. " 
                "I'll post the findings here when the review is complete."
            )
        }

        async with httpx.AsyncClient() as client: 
            response = await client.post(url, headers=headers, json=payload) 
            response.raise_for_status() 
            return response.json()

    async def post_review(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        token: str,
        commit_id: str,
        findings: list,
    ) -> dict:
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2026-03-10",
        }

        url = (
            f"{GITHUB_API}/repos/"
            f"{owner}/{repo}/pulls/"
            f"{pr_number}/reviews"
        )

        comments = []

        for finding in findings:
            comments.append({
                "path": finding.path,
                "line": finding.line,
                "side": "RIGHT",
                "body": (
                    f"**{finding.severity.upper()}: "
                    f"{finding.title}**\n\n"
                    f"{finding.body}"
                ),
            })

        if findings:
            body = (
                f"🤖 **AI review complete.** "
                f"I found {len(findings)} "
                f"potential issue"
                f"{'s' if len(findings) != 1 else ''}."
            )
        else:
            body = (
                "🤖 **AI review complete.**\n\n"
                "I didn't find any significant issues "
                "in the changes."
            )

        payload = {
            "commit_id": commit_id,
            "body": body,
            "event": "COMMENT",
            "comments": comments,
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                headers=headers,
                json=payload,
            )

        response.raise_for_status()

        return response.json()

    async def get_pr_head_sha(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        token: str,
    ) -> str:
        url = (
            f"https://api.github.com/repos/"
            f"{owner}/{repo}/pulls/{pr_number}"
        )

        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2026-03-10",
        }

        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(
                url,
                headers=headers,
            )

            response.raise_for_status()

            data = response.json()

            return data["head"]["sha"]

    async def get_file_content(
        self,
        owner: str,
        repo: str,
        path: str,
        token: str,
        ref: str,
    ) -> str:
        url = (
            f"https://api.github.com/repos/"
            f"{owner}/{repo}/contents/{path}"
        )

        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2026-03-10",
        }

        async with httpx.AsyncClient() as client:
            response = await client.get(
            url,
            headers=headers,
            params={"ref": ref},
        )

        response.raise_for_status()

        data = response.json()

        if data.get("encoding") != "base64":
            raise RuntimeError(
                f"Unexpected encoding for {path}: "
                f"{data.get('encoding')}"
            )

        import base64

        return base64.b64decode(
            data["content"]
        ).decode("utf-8")