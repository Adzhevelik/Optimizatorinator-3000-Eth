import asyncio
import sys
import pickle
import numpy as np
import aiohttp
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional

sys.path.append(str(Path(__file__).parent.parent / 'backend'))


RPC_ENDPOINTS = [
    "https://ethereum.publicnode.com",
    "https://rpc.ankr.com/eth",
    "https://eth.llamarpc.com",
    "https://cloudflare-eth.com",
]

NETWORK_CONFIG = {
    'blocks_per_hour':    300,
    'days':               90,
    'step':               2,
    'batch_size':         6,
    'sleep':              0.2,
    'checkpoint_every':   5000,
}

MISSING_WARN_THRESHOLD  = 0.05
MISSING_RETRY_THRESHOLD = 0.20
MAX_RETRIES             = 3


async def test_rpc_endpoint(session: aiohttp.ClientSession, url: str) -> Optional[int]:
    payload = {"jsonrpc": "2.0", "method": "eth_blockNumber", "params": [], "id": 1}
    try:
        async with session.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=aiohttp.ClientTimeout(total=10),
        ) as r:
            if r.content_type != 'application/json':
                return None
            data = await r.json()
            if 'result' not in data:
                return None
            return int(data['result'], 16)
    except Exception:
        return None


async def find_active_endpoint(session: aiohttp.ClientSession) -> tuple[str, int]:
    for endpoint in RPC_ENDPOINTS:
        result = await test_rpc_endpoint(session, endpoint)
        if result:
            print(f"  Connected: {endpoint} (block #{result:,})")
            return endpoint, result
        print(f"  Failed:    {endpoint}")
    raise RuntimeError("All RPC endpoints failed")


async def get_block_rpc(
    session: aiohttp.ClientSession,
    rpc_url: str,
    block_number: int,
    retries: int = 3,
) -> Optional[dict]:
    payload = {
        "jsonrpc": "2.0",
        "method":  "eth_getBlockByNumber",
        "params":  [hex(block_number), True],
        "id":      block_number,
    }
    for attempt in range(retries):
        try:
            async with session.post(
                rpc_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=45),
            ) as r:
                if r.content_type != 'application/json':
                    continue
                data = await r.json()
                result = data.get('result')
                if result:
                    return result
        except Exception:
            if attempt < retries - 1:
                await asyncio.sleep(0.8 * (attempt + 1))
    return None


def _parse_block(block_data: dict) -> Optional[dict]:
    try:
        gas_used  = int(block_data.get('gasUsed',  '0x0'), 16)
        gas_limit = int(block_data.get('gasLimit', '0x0'), 16)
        timestamp = int(block_data.get('timestamp', '0x0'), 16)
        block_num = int(block_data.get('number',   '0x0'), 16)
        base_fee  = (
            int(block_data['baseFeePerGas'], 16)
            if 'baseFeePerGas' in block_data else 0
        )

        if base_fee <= 0 or gas_limit <= 0:
            return None

        transactions = block_data.get('transactions', [])
        tx_count = len(transactions)

        tips = []
        type2_count = 0
        for tx in transactions:
            if not isinstance(tx, dict):
                continue
            tx_type = int(tx.get('type', '0x0'), 16)
            if tx_type == 2:
                type2_count += 1
                max_fee = int(tx.get('maxFeePerGas', '0x0'), 16)
                max_tip = int(tx.get('maxPriorityFeePerGas', '0x0'), 16)
                if max_fee >= base_fee and max_tip > 0:
                    effective_tip = min(max_tip, max_fee - base_fee)
                    if effective_tip > 0:
                        tips.append(effective_tip)

        median_tip_gwei = float(np.median(tips)) / 1e9 if tips else 0.0
        mean_tip_gwei   = float(np.mean(tips))   / 1e9 if tips else 0.0
        tip_p75_gwei    = float(np.percentile(tips, 75)) / 1e9 if len(tips) >= 4 else 0.0
        tip_p25_gwei    = float(np.percentile(tips, 25)) / 1e9 if len(tips) >= 4 else 0.0
        type2_ratio     = type2_count / tx_count if tx_count > 0 else 0.0

        return {
            'network':           'ethereum',
            'number':            block_num,
            'timestamp':         timestamp,
            'datetime':          pd.Timestamp(timestamp, unit='s'),
            'gas_used':          gas_used,
            'gas_limit':         gas_limit,
            'base_fee_per_gas':  base_fee,
            'transaction_count': tx_count,
            'gas_price_gwei':    base_fee / 1e9,
            'utilization':       gas_used / gas_limit,
            'median_tip_gwei':   median_tip_gwei,
            'mean_tip_gwei':     mean_tip_gwei,
            'tip_p75_gwei':      tip_p75_gwei,
            'tip_p25_gwei':      tip_p25_gwei,
            'tip_tx_count':      len(tips),
            'type2_ratio':       type2_ratio,
            'source':            'rpc',
        }
    except Exception:
        return None


def _load_checkpoint(checkpoint_path: Path) -> tuple[list, set, Optional[int]]:
    if not checkpoint_path.exists():
        return [], set(), None
    try:
        with open(checkpoint_path, 'rb') as f:
            data = pickle.load(f)
        if isinstance(data, dict):
            blocks = data['blocks']
            start_block = data.get('start_block')
        else:
            blocks = data
            start_block = min(b['number'] for b in blocks) if blocks else None
        collected_numbers = {b['number'] for b in blocks}
        print(f"  Checkpoint loaded: {len(blocks):,} blocks already collected")
        return blocks, collected_numbers, start_block
    except Exception as exc:
        print(f"  WARNING: failed to load checkpoint: {exc}")
        return [], set(), None


def _save_checkpoint(blocks: list, checkpoint_path: Path, start_block: int) -> None:
    if not blocks:
        return
    try:
        with open(checkpoint_path, 'wb') as f:
            pickle.dump({'blocks': blocks, 'start_block': start_block}, f)
    except Exception as exc:
        print(f"  WARNING: checkpoint save failed: {exc}")


def _migrate_parquet_checkpoint(data_path: Path) -> None:
    old_path = data_path / 'ethereum_checkpoint.parquet'
    new_path = data_path / 'ethereum_checkpoint.pkl'
    if not old_path.exists() or new_path.exists():
        return
    try:
        df_ck = pd.read_parquet(old_path)
        blocks_ck = df_ck.to_dict('records')
        start_block = min(b['number'] for b in blocks_ck) if blocks_ck else None
        with open(new_path, 'wb') as f:
            pickle.dump({'blocks': blocks_ck, 'start_block': start_block}, f)
        old_path.unlink()
        print(f"  Migrated parquet checkpoint: {len(blocks_ck):,} blocks -> pkl")
    except Exception as exc:
        print(f"  WARNING: checkpoint migration failed: {exc}")


async def _fetch_blocks(
    session: aiohttp.ClientSession,
    rpc_url: str,
    block_numbers: list,
    batch_size: int,
    sleep_time: float,
    checkpoint_path: Path,
    existing_blocks: list,
    checkpoint_every: int,
    start_block: int,
    label: str = '',
) -> tuple[list, int, str]:
    blocks = list(existing_blocks)
    errors = 0
    current_rpc = rpc_url
    prefix = f"[{label}] " if label else ""

    for i in range(0, len(block_numbers), batch_size):
        batch = block_numbers[i:i + batch_size]
        tasks = [get_block_rpc(session, current_rpc, num) for num in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        batch_errors = sum(
            1 for r in results if isinstance(r, Exception) or not r
        )
        if batch_errors == len(batch):
            print(f"  {prefix}Full batch failure — rotating endpoint...")
            for endpoint in RPC_ENDPOINTS:
                if endpoint != current_rpc:
                    ok = await test_rpc_endpoint(session, endpoint)
                    if ok:
                        print(f"  {prefix}Switched to: {endpoint}")
                        current_rpc = endpoint
                        await asyncio.sleep(1.0)
                        break

        for block_data in results:
            if isinstance(block_data, Exception) or not block_data:
                errors += 1
                continue
            parsed = _parse_block(block_data)
            if parsed:
                blocks.append(parsed)
            else:
                errors += 1

        blocks_done = i + batch_size
        if blocks_done % checkpoint_every == 0 or blocks_done >= len(block_numbers):
            _save_checkpoint(blocks, checkpoint_path, start_block)
            pct = min(blocks_done / len(block_numbers) * 100, 100)
            eta_batches = max(0, (len(block_numbers) - blocks_done) // batch_size)
            eta_min = eta_batches * (sleep_time + 0.65) / 60
            print(
                f"  {prefix}{len(blocks):,} blocks | "
                f"{pct:.1f}% | errors: {errors} | "
                f"ETA: ~{eta_min:.0f} min"
            )

        await asyncio.sleep(sleep_time)

    return blocks, errors, current_rpc


def _check_missing(
    collected_numbers: set, expected_numbers: list
) -> tuple[set, float]:
    expected_set = set(expected_numbers)
    missing = expected_set - collected_numbers
    missing_rate = len(missing) / len(expected_set)
    return missing, missing_rate


async def collect_3_months() -> pd.DataFrame:
    cfg            = NETWORK_CONFIG
    days           = cfg['days']
    bph            = cfg['blocks_per_hour']
    step           = cfg['step']
    batch_size     = cfg['batch_size']
    sleep_time     = cfg['sleep']
    checkpoint_n   = cfg['checkpoint_every']

    total_blocks    = days * 24 * bph
    blocks_to_fetch = total_blocks // step

    data_path = Path(__file__).parent.parent / 'data'
    data_path.mkdir(exist_ok=True)
    checkpoint_path = data_path / 'ethereum_checkpoint.pkl'

    _migrate_parquet_checkpoint(data_path)

    print(f"\n{'='*60}")
    print(f"  Target : {blocks_to_fetch:,} blocks (step={step}, {days} days)")
    print(f"  Est.   : ~{blocks_to_fetch / batch_size * (sleep_time + 0.65) / 3600:.1f} hours")
    print(f"  Chkpt  : {checkpoint_path.name}")
    print(f"{'='*60}")

    existing_blocks, existing_numbers, saved_start = _load_checkpoint(checkpoint_path)

    connector = aiohttp.TCPConnector(limit=10, force_close=True)
    async with aiohttp.ClientSession(connector=connector) as session:
        rpc_url, latest = await find_active_endpoint(session)

        if saved_start is not None:
            start_block = saved_start
        else:
            start_block = latest - total_blocks

        block_numbers = list(range(start_block, start_block + total_blocks, step))

        remaining = [n for n in block_numbers if n not in existing_numbers]
        print(f"  To fetch: {len(remaining):,} blocks ({len(existing_numbers):,} already in checkpoint)")

        if remaining:
            blocks, errors, rpc_url = await _fetch_blocks(
                session, rpc_url, remaining,
                batch_size, sleep_time,
                checkpoint_path, existing_blocks,
                checkpoint_every=checkpoint_n,
                start_block=start_block,
            )
        else:
            blocks = existing_blocks
            errors = 0

        df = pd.DataFrame(blocks)
        if df.empty:
            raise RuntimeError("No blocks collected")

        df = df.drop_duplicates(subset=['number'])
        collected_numbers = set(df['number'].tolist())
        missing, missing_rate = _check_missing(collected_numbers, block_numbers)

        print(f"\n  Collected : {len(df):,} / {blocks_to_fetch:,} blocks")
        print(f"  Errors    : {errors}")
        print(f"  Missing   : {len(missing):,} ({missing_rate * 100:.2f}%)")

        if MISSING_WARN_THRESHOLD <= missing_rate < MISSING_RETRY_THRESHOLD:
            print(f"  WARNING: {missing_rate*100:.1f}% missing — retrying...")
            missing_list = sorted(missing)
            for attempt in range(1, MAX_RETRIES + 1):
                print(f"  Retry {attempt}/{MAX_RETRIES} ({len(missing_list):,} blocks)...")
                await asyncio.sleep(2)
                retry_blocks, _, rpc_url = await _fetch_blocks(
                    session, rpc_url, missing_list,
                    batch_size=3, sleep_time=0.4,
                    checkpoint_path=checkpoint_path,
                    existing_blocks=blocks,
                    checkpoint_every=1000,
                    start_block=start_block,
                    label=f"retry {attempt}",
                )
                if retry_blocks:
                    df = (
                        pd.DataFrame(retry_blocks)
                        .drop_duplicates(subset=['number'])
                    )
                    collected_numbers = set(df['number'].tolist())
                    missing, missing_rate = _check_missing(collected_numbers, block_numbers)
                    missing_list = sorted(missing)
                    blocks = retry_blocks
                    print(f"  After retry {attempt}: missing {len(missing):,} ({missing_rate*100:.2f}%)")
                if missing_rate < MISSING_WARN_THRESHOLD:
                    break

        elif missing_rate >= MISSING_RETRY_THRESHOLD:
            raise RuntimeError(
                f"Missing rate {missing_rate*100:.1f}% > 20% — "
                "data quality too low. Check RPC endpoints."
            )

        df = df.sort_values('datetime').reset_index(drop=True)

        out_path = data_path / f"ethereum_3m_{datetime.utcnow().strftime('%Y%m%d')}.csv"
        df.to_csv(out_path, index=False)
        print(f"\n  Saved -> {out_path.name}")

        if checkpoint_path.exists():
            checkpoint_path.unlink()
            print(f"  Checkpoint removed")

        return df


if __name__ == "__main__":
    try:
        asyncio.run(collect_3_months())
    except KeyboardInterrupt:
        print("\n  Interrupted — checkpoint preserved, resume by re-running")
    except Exception as exc:
        print(f"\n  FATAL: {exc}")
        raise
    