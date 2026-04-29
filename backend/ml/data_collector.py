import pandas as pd
import asyncio
from datetime import datetime
from typing import List, Dict
import logging
from pathlib import Path

from core.network_client import BlockchainClient
from config import settings

logger = logging.getLogger(__name__)


class DataCollector:

    def __init__(self):
        self.client    = BlockchainClient()
        self.data_path = Path(settings.DATA_PATH)
        self.data_path.mkdir(exist_ok=True)

    async def collect_historical_data(self, blocks_count: int = 10000) -> pd.DataFrame:
        logger.info(f"Starting historical data collection for ethereum")
        print(f"\n=== Collecting {blocks_count} blocks from ethereum ===")

        latest_block = await self.client.get_latest_block_number('ethereum')
        print(f"ethereum: Latest block number: {latest_block}")

        blocks = await self.client.collect_block_range('ethereum', latest_block, blocks_count)

        df = self._process_blocks_to_dataframe(blocks)

        output_file = self.data_path / f"ethereum_historical_{datetime.now().strftime('%Y%m%d')}.csv"
        df.to_csv(output_file, index=False)

        logger.info(f"Collected {len(df)} blocks for ethereum")
        print(f"ethereum: Saved {len(df)} blocks to {output_file}")

        return df

    def _process_blocks_to_dataframe(self, blocks: List[Dict]) -> pd.DataFrame:
        df = pd.DataFrame(blocks)
        df['network']        = 'ethereum'
        df['datetime']       = pd.to_datetime(df['timestamp'], unit='s')
        df['utilization']    = df['gas_used'] / df['gas_limit']
        df['gas_price_gwei'] = df['base_fee_per_gas'] / 1e9
        return df

    async def update_incremental(self) -> pd.DataFrame:
        latest_file = self._get_latest_dataset()

        if latest_file.exists():
            existing_data = pd.read_csv(latest_file)
            last_block    = existing_data['number'].max()
        else:
            last_block = await self.client.get_latest_block_number('ethereum') - 1000

        current_block    = await self.client.get_latest_block_number('ethereum')
        new_blocks_count = current_block - last_block

        if new_blocks_count > 0:
            new_blocks = await self.client.collect_block_range('ethereum', current_block, new_blocks_count)
            new_df     = self._process_blocks_to_dataframe(new_blocks)

            if latest_file.exists():
                existing_data = pd.read_csv(latest_file)
                combined      = pd.concat([existing_data, new_df], ignore_index=True)
                combined.to_csv(latest_file, index=False)
            else:
                new_df.to_csv(latest_file, index=False)

            return new_df

        return pd.DataFrame()

    def _get_latest_dataset(self) -> Path:
        files = sorted(self.data_path.glob('ethereum_historical_*.csv'), reverse=True)
        if files:
            return files[0]
        return self.data_path / 'ethereum_historical.csv'
    