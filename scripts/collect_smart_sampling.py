import asyncio
import pickle
import numpy as np
import aiohttp
import pandas as pd
from pathlib import Path
from datetime import datetime


ALCHEMY_KEYS = [
    "https://eth-mainnet.g.alchemy.com/v2/THdHqdXN-d1qMDCMb41iZ",
    "https://eth-mainnet.g.alchemy.com/v2/C9sJr1KVkmNQIG2EfS4T3",
    "https://eth-mainnet.g.alchemy.com/v2/Z4AidgRCTQJB_GdGSA41b",
    "https://eth-mainnet.g.alchemy.com/v2/I7zgClEE3x3__jTBJVceI",
    "https://eth-mainnet.g.alchemy.com/v2/q7wVvNAW1Ut3n1IRLtrys"
]

EIP1559_BLOCK    = 12_965_000
BATCH_SIZE       = 25
SLEEP            = 1.0
CHECKPOINT_EVERY = 5000

MISSING_WARN_THRESHOLD  = 0.05
MISSING_RETRY_THRESHOLD = 0.20
MAX_RETRIES             = 3


class RateLimitExhausted(Exception):
    pass


async def get_latest_block(session: aiohttp.ClientSession, url: str) -> int:
    payload = {"jsonrpc": "2.0", "id": 1, "method": "eth_blockNumber", "params": []}
    async with session.post(url, json=payload) as resp:
        data = await resp.json()
        return int(data["result"], 16)


async def fetch_batch(
    session: aiohttp.ClientSession,
    url: str,
    block_numbers: list,
    retries: int = 5,
) -> list:
    payload = [
        {
            "jsonrpc": "2.0",
            "id":      i,
            "method":  "eth_getBlockByNumber",
            "params":  [hex(num), True],
        }
        for i, num in enumerate(block_numbers)
    ]

    consecutive_429 = 0
    for attempt in range(retries):
        try:
            async with session.post(url, json=payload) as resp:
                results = await resp.json()
                if not isinstance(results, list):
                    raise ValueError(f"Non-list response: {results}")
                errors = [r for r in results if "error" in r]
                if errors:
                    code = errors[0]["error"].get("code")
                    if code == 429:
                        consecutive_429 += 1
                        if consecutive_429 >= 3:
                            raise RateLimitExhausted(f"Key {url[-6:]} exhausted")
                        wait = 2.0 * (attempt + 1)
                        print(f"  [{url[-6:]}] 429 rate limit, waiting {wait}s...")
                        await asyncio.sleep(wait)
                        continue
                consecutive_429 = 0
                return [r.get("result") for r in sorted(results, key=lambda x: x.get("id", 0))]
        except RateLimitExhausted:
            raise
        except Exception as e:
            print(f"  [{url[-6:]}] Batch attempt {attempt + 1} failed: {e}")
            await asyncio.sleep(1.0 * (attempt + 1))

    return [None] * len(block_numbers)


def parse_block(block_data: dict) -> dict | None:
    try:
        gas_used  = int(block_data.get("gasUsed",  "0x0"), 16)
        gas_limit = int(block_data.get("gasLimit", "0x0"), 16)
        timestamp = int(block_data.get("timestamp","0x0"), 16)
        block_num = int(block_data.get("number",   "0x0"), 16)
        base_fee  = int(block_data["baseFeePerGas"], 16) if "baseFeePerGas" in block_data else 0

        if base_fee <= 0 or gas_limit <= 0:
            return None

        transactions = block_data.get("transactions", [])
        tx_count = len(transactions)

        tips = []
        type2_count = 0
        for tx in transactions:
            if not isinstance(tx, dict):
                continue
            tx_type = int(tx.get("type", "0x0"), 16)
            if tx_type == 2:
                type2_count += 1
                max_fee = int(tx.get("maxFeePerGas", "0x0"), 16)
                max_tip = int(tx.get("maxPriorityFeePerGas", "0x0"), 16)
                if max_fee >= base_fee and max_tip > 0:
                    effective_tip = min(max_tip, max_fee - base_fee)
                    if effective_tip > 0:
                        tips.append(effective_tip)

        return {
            "network":           "ethereum",
            "number":            block_num,
            "timestamp":         timestamp,
            "datetime":          pd.Timestamp(timestamp, unit="s"),
            "gas_used":          gas_used,
            "gas_limit":         gas_limit,
            "base_fee_per_gas":  base_fee,
            "transaction_count": tx_count,
            "gas_price_gwei":    base_fee / 1e9,
            "utilization":       gas_used / gas_limit,
            "median_tip_gwei":   float(np.median(tips)) / 1e9 if tips else 0.0,
            "mean_tip_gwei":     float(np.mean(tips))   / 1e9 if tips else 0.0,
            "tip_p75_gwei":      float(np.percentile(tips, 75)) / 1e9 if len(tips) >= 4 else 0.0,
            "tip_p25_gwei":      float(np.percentile(tips, 25)) / 1e9 if len(tips) >= 4 else 0.0,
            "tip_tx_count":      len(tips),
            "type2_ratio":       type2_count / tx_count if tx_count > 0 else 0.0,
            "source":            "alchemy",
        }
    except Exception:
        return None


def load_checkpoint(path: Path) -> tuple[list, set, int | None]:
    if not path.exists():
        return [], set(), None
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        blocks     = data["blocks"]
        start      = data.get("start_block")
        last       = max((b["number"] for b in blocks), default=None)
        print(f"  Checkpoint: {len(blocks):,} blocks | last block: {last}")
        return blocks, {b["number"] for b in blocks}, start
    except Exception as e:
        print(f"  WARNING: checkpoint load failed: {e}")
        return [], set(), None


def save_checkpoint(blocks: list, path: Path, start_block: int) -> None:
    if not blocks:
        return
    try:
        with open(path, "wb") as f:
            pickle.dump({"blocks": blocks, "start_block": start_block}, f)
    except Exception as e:
        print(f"  WARNING: checkpoint save failed: {e}")


def check_missing(collected: set, expected: list) -> tuple[set, float]:
    expected_set = set(expected)
    missing = expected_set - collected
    return missing, len(missing) / len(expected_set)


async def collect_part(
    part_index:    int,
    alchemy_url:   str,
    block_numbers: list,
    data_path:     Path,
) -> list:
    label           = f"part{part_index + 1}"
    checkpoint_path = data_path / f"ethereum_checkpoint_{label}.pkl"

    existing_blocks, existing_numbers, _ = load_checkpoint(checkpoint_path)
    remaining = [n for n in block_numbers if n not in existing_numbers]
    last_saved = max(existing_numbers) if existing_numbers else None

    print(f"  [{label}] {len(block_numbers):,} blocks total | {len(remaining):,} to fetch | last saved: {last_saved}")

    blocks = list(existing_blocks)
    errors = 0

    connector = aiohttp.TCPConnector(limit=10, force_close=True)
    async with aiohttp.ClientSession(connector=connector) as session:
        for i in range(0, len(remaining), BATCH_SIZE):
            batch_nums = remaining[i:i + BATCH_SIZE]

            try:
                results = await fetch_batch(session, alchemy_url, batch_nums)
            except RateLimitExhausted:
                save_checkpoint(blocks, checkpoint_path, block_numbers[0])
                last = max((b["number"] for b in blocks), default=None)
                print(f"  [{label}] Monthly limit exhausted — checkpoint saved | last block: {last}")
                return blocks

            for raw in results:
                if not raw:
                    errors += 1
                    continue
                parsed = parse_block(raw)
                if parsed:
                    blocks.append(parsed)
                else:
                    errors += 1

            done = i + len(batch_nums)
            if done % CHECKPOINT_EVERY == 0 or done >= len(remaining):
                save_checkpoint(blocks, checkpoint_path, block_numbers[0])
                last    = max((b["number"] for b in blocks), default=None)
                pct     = done / len(remaining) * 100
                eta_min = max(0, len(remaining) - done) / BATCH_SIZE * SLEEP / 60
                print(
                    f"  [{label}] {len(blocks):,} blocks | {pct:.1f}% | "
                    f"last block: {last} | errors: {errors} | ETA: ~{eta_min:.0f} min"
                )

            await asyncio.sleep(SLEEP)

    if checkpoint_path.exists():
        checkpoint_path.unlink()

    return blocks


async def collect() -> pd.DataFrame:
    data_path = Path(__file__).parent.parent / "data"
    data_path.mkdir(exist_ok=True)

    connector = aiohttp.TCPConnector(limit=5)
    async with aiohttp.ClientSession(connector=connector) as session:
        latest = await get_latest_block(session, ALCHEMY_KEYS[0])

    print(f"  Latest block : {latest:,}")
    print(f"  EIP-1559 from: {EIP1559_BLOCK:,}")

    all_blocks = list(range(EIP1559_BLOCK, latest))
    n          = len(all_blocks)
    part_size  = n // len(ALCHEMY_KEYS)

    parts = [
        all_blocks[i * part_size: (i + 1) * part_size]
        for i in range(len(ALCHEMY_KEYS))
    ]
    parts[-1].extend(all_blocks[len(ALCHEMY_KEYS) * part_size:])

    eta_min = part_size / BATCH_SIZE * SLEEP / 60
    print(f"\n{'='*60}")
    print(f"  Total blocks : {n:,}")
    print(f"  Parts        : {len(parts)} × ~{part_size:,} blocks")
    print(f"  ETA per part : ~{eta_min:.0f} min (~{eta_min/60:.1f} hours)")
    print(f"{'='*60}\n")

    tasks = [
        collect_part(i, ALCHEMY_KEYS[i], parts[i], data_path)
        for i in range(len(ALCHEMY_KEYS))
    ]
    results = await asyncio.gather(*tasks)

    all_collected = [b for part_blocks in results for b in part_blocks]

    df = (
        pd.DataFrame(all_collected)
        .drop_duplicates(subset=["number"])
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    missing, missing_rate = check_missing(set(df["number"].tolist()), all_blocks)
    print(f"\n  Collected : {len(df):,} / {n:,} blocks")
    print(f"  Missing   : {len(missing):,} ({missing_rate * 100:.2f}%)")

    if missing_rate >= MISSING_RETRY_THRESHOLD:
        print(f"  WARNING: missing rate {missing_rate*100:.1f}% > 20% — checkpoints preserved for resume")
        return df

    out = data_path / f"ethereum_historical_{datetime.utcnow().strftime('%Y%m%d')}.csv"
    df.to_csv(out, index=False)
    print(f"\n  Saved -> {out.name}  ({len(df):,} rows)")

    return df


if __name__ == "__main__":
    try:
        asyncio.run(collect())
    except KeyboardInterrupt:
        print("\n  Interrupted — checkpoints preserved, resume by re-running")
    except Exception as e:
        print(f"\n  FATAL: {e}")
        raise
