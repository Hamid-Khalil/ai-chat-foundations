import asyncio

import httpx


async def main():
    base = "http://127.0.0.1:8000/api/v1"
    async with httpx.AsyncClient() as c:
        r = await c.post(f"{base}/conversations", json={"title": "Concurrency"})
        cid = r.json()["id"]
        tasks = [
            c.post(
                f"{base}/conversations/{cid}/chat",
                json={"content": f"msg {i}", "mode": "chat"},
            )
            for i in range(5)
        ]
        await asyncio.gather(*tasks)
        msgs = (await c.get(f"{base}/conversations/{cid}/messages")).json()["items"]
        assert len(msgs) == 10
        print("Concurrency check OK: exactly 10 stored.")


if __name__ == "__main__":
    asyncio.run(main())
