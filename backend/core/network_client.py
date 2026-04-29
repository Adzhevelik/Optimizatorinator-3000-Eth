import aiohttp
import asyncio
from typing import Dict, List, Optional
import logging

from config import settings

logger = logging.getLogger(__name__)


class BlockchainClient:

    def __init__(self):
        self.alchemy_url        = f"https://eth-mainnet.g.alchemy.com/v2/{settings.ALCHEMY_API_KEY}"
        self.etherscan_base_url = "https://api.etherscan.io/v2/api"
        self.etherscan_api_key  = settings.ETHERSCAN_API_KEY
        self.session            = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def _get_session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session

    async def _rpc(self, method: str, params: list):
        payload = {
            "jsonrpc": "2.0",
            "id":      1,
            "method":  method,
            "params":  params,
        }
        session = await self._get_session()
        async with session.post(self.alchemy_url, json=payload) as resp:
            data = await resp.json()
            if "error" in data:
                raise ValueError(f"Alchemy RPC error: {data['error']}")
            return data.get("result")

    async def _rpc_batch(self, calls: list) -> list:
        payload = [
            {"jsonrpc": "2.0", "id": i, "method": call["method"], "params": call["params"]}
            for i, call in enumerate(calls)
        ]
        session = await self._get_session()
        async with session.post(self.alchemy_url, json=payload) as resp:
            results = await resp.json()
            if not isinstance(results, list):
                logger.error(f"Alchemy batch returned non-list: {results}")
                return []
            errors = [r for r in results if "error" in r]
            if errors:
                logger.warning(f"Batch had {len(errors)} errors: {errors[0]}")
            return [r.get("result") for r in sorted(results, key=lambda x: x.get("id", 0))]

    async def get_latest_block_number(self, network: str = "ethereum") -> int:
        result = await self._rpc("eth_blockNumber", [])
        return int(result, 16)

    async def get_block_details(self, block_number: int) -> Optional[Dict]:
        result = await self._rpc(
            "eth_getBlockByNumber",
            [hex(block_number), False],
        )
        if not result:
            return None
        try:
            return {
                "number":            block_number,
                "timestamp":         int(result["timestamp"], 16),
                "gas_used":          int(result["gasUsed"], 16),
                "gas_limit":         int(result["gasLimit"], 16),
                "base_fee_per_gas":  int(result.get("baseFeePerGas", "0x0"), 16),
                "transaction_count": len(result.get("transactions", [])),
            }
        except Exception as e:
            logger.error(f"Error parsing block {block_number}: {e}")
            return None

    async def collect_block_range(
        self, network: str = "ethereum", start_block: int = 0, count: int = 100
    ) -> List[Dict]:
        blocks = []
        logger.info(f"ethereum: collecting {count} blocks from {start_block}")

        BATCH = 25

        for i in range(0, count, BATCH):
            batch_size = min(BATCH, count - i)

            calls = [
                {
                    "method": "eth_getBlockByNumber",
                    "params": [hex(start_block - i - j), False],
                }
                for j in range(batch_size)
            ]

            results = None
            for attempt in range(5):
                try:
                    results = await self._rpc_batch(calls)
                    if results and any(r is not None for r in results):
                        break
                except Exception as e:
                    logger.warning(f"Batch attempt {attempt + 1} failed: {e}")
                wait = 1.0 * (attempt + 1)
                logger.warning(f"Rate limited, waiting {wait}s before retry")
                await asyncio.sleep(wait)

            if not results:
                logger.error(f"Batch at offset {i} failed after 5 attempts, skipping")
                continue

            for j, result in enumerate(results):
                if not result:
                    continue
                try:
                    blocks.append({
                        "number":            start_block - i - j,
                        "timestamp":         int(result["timestamp"], 16),
                        "gas_used":          int(result["gasUsed"], 16),
                        "gas_limit":         int(result["gasLimit"], 16),
                        "base_fee_per_gas":  int(result.get("baseFeePerGas", "0x0"), 16),
                        "transaction_count": len(result.get("transactions", [])),
                    })
                except Exception as e:
                    logger.error(f"Error parsing block at offset {j}: {e}")

            print(f"ethereum: {len(blocks)}/{count} blocks collected")
            await asyncio.sleep(1)

        return blocks

    async def get_gas_oracle(self, network: str = "ethereum") -> Dict:
        params = {
            "chainid": "1",
            "module":  "gastracker",
            "action":  "gasoracle",
            "apikey":  self.etherscan_api_key,
        }
        try:
            session = await self._get_session()
            async with session.get(self.etherscan_base_url, params=params) as resp:
                data = await resp.json()
                if data.get("status") == "1" and "result" in data:
                    r = data["result"]
                    return {
                        "safe_gas_price":    float(r.get("SafeGasPrice",    10)),
                        "propose_gas_price": float(r.get("ProposeGasPrice", 15)),
                        "fast_gas_price":    float(r.get("FastGasPrice",    20)),
                    }
        except Exception as e:
            logger.warning(f"Gas oracle failed: {e}")

        return {"safe_gas_price": 10.0, "propose_gas_price": 15.0, "fast_gas_price": 20.0}

    async def get_pending_count(self) -> int:
        try:
            result = await self._rpc(
                "eth_getBlockTransactionCountByNumber", ["pending"]
            )
            if result:
                return int(result, 16)
        except Exception:
            pass
        return 0

    async def get_token_price_usd(self, network: str = "ethereum") -> Optional[float]:
        return settings.TOKEN_PRICE

    async def close(self):
        if self.session:
            await self.session.close()
            self.session = None
