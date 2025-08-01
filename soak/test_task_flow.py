import asyncio
import logging
import time

from soak.async_decorators import flow, task, cancel_flow

logging.basicConfig(level=logging.INFO)


@task
async def sleeper(name: str, delay: float):
    print(f"[{time.time():.2f}] {name} starting (sleep {delay}s)")
    await asyncio.sleep(delay)
    print(f"[{time.time():.2f}] {name} done")
    return name


@flow
async def run_parallel_tasks():
    t0 = time.time()
    print(f"[{t0:.2f}] Flow started")

    tasks = [
        asyncio.create_task(sleeper(f"task_{i}", delay)) for i, delay in enumerate([1, 2, 3, 4, 5])
    ]

    ctx = asyncio.get_running_loop().create_task(cancel_soon(3.5))  # cancel after 1.5s

    try:
        results = await asyncio.gather(*tasks)
        print(f"[{time.time():.2f}] Flow completed with results: {results}")
    except asyncio.CancelledError:
        print(f"[{time.time():.2f}] Flow was cancelled")
    except Exception as e:
        print(f"[{time.time():.2f}] Flow error: {e}")


async def cancel_soon(delay):
    await asyncio.sleep(delay)
    print(f"[{time.time():.2f}] >>> Calling cancel_flow()")
    cancel_flow()


def main():
    print(">>> Running flow with cancellable parallel tasks")
    try:
        run_parallel_tasks.sync()
    except Exception as e:
        print(f"Flow exception: {e}")


if __name__ == "__main__":
    main()
