import argparse
import asyncio
import yaml

from enhanced_csp.network import EnhancedCSPNetwork
from enhanced_csp.network.core.types import NetworkConfig


def load_config(path: str) -> NetworkConfig:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return NetworkConfig(**data.get("network", data))


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    network = EnhancedCSPNetwork(config)
    await network.start()

    try:
        while True:
            await asyncio.sleep(60)
    except KeyboardInterrupt:
        await network.stop()


if __name__ == "__main__":
    asyncio.run(main())
