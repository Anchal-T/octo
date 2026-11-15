import asyncio

from hypercorn.asyncio import serve
from hypercorn.config import Config

from .main import build_runtime, close_runtime, create_app
from .settings import GithubSettings


async def main() -> None:
    settings = GithubSettings.from_env()
    settings.validate_for_start()
    runtime = build_runtime(settings)
    config = Config()
    config.bind = ["0.0.0.0:8003"]
    try:
        await serve(create_app(runtime), config)
    finally:
        await close_runtime(runtime)


if __name__ == "__main__":
    asyncio.run(main())
