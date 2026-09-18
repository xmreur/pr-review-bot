
import asyncio

from app.github import GithubClient
from app.queue import ReviewJob, enqueue_review, process_review


async def main():
    github = GithubClient()

    installation_id = await github.get_installation_id(
        "xmreur",
        "neoorm",
    )

    print("Installation ID:", installation_id)

    token = await github.create_installation_token(
        installation_id
    )

    print("Got installation token")

    files = await github.get_pr_files(
        "xmreur",
        "neoorm",
        182,
        token,
    )

   
    await github.post_confirm("xmreur", "neoorm", 182, token)

    await process_review(
        ReviewJob(
            owner="xmreur",
            repo="neoorm",
            pr_number=182,
            installation_id=installation_id,
            commit_sha="d1cbffa86c3bc70b8f24337d1a5b212112252761",
        )
    )
    


if __name__ == "__main__":
    asyncio.run(main())
