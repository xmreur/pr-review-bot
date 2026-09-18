
import asyncio

from app.reviewer import review_diff


TEST_DIFF = """
--- FILE: app.py ---
@@ -1,5 +1,6 @@

 def divide(a, b):
+    print("dividing")
     return a / b
"""


async def main():
    review = await review_diff(TEST_DIFF)

    print(
        review.model_dump_json(indent=2)
    )


if __name__ == "__main__":
    asyncio.run(main())
