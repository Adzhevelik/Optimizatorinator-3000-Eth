from pydantic_settings import BaseSettings
from pydantic import ConfigDict

class Settings(BaseSettings):
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ETHERSCAN_API_KEY: str = ""
    ALCHEMY_API_KEY:   str = ""
    MODEL_PATH: str = "models/"
    DATA_PATH:  str = "data/"
    COLLECTION_INTERVAL_HOURS: int = 1
    BLOCKS_PER_COLLECTION:     int = 100
    ETHEREUM_RPC_ENDPOINTS: list = [
        "https://rpc.ankr.com/eth",
        "https://eth.llamarpc.com",
        "https://ethereum.publicnode.com",
    ]
    RETRAIN_INTERVAL_HOURS:          int   = 24
    MIN_BLOCKS_FOR_TRAINING:         int   = 500
    TEST_SIZE_BLOCKS:                int   = 50
    TEMPORAL_WEIGHT_HALFLIFE_DAYS:   int   = 30
    FRESH_DATA_WEIGHT:               float = 2.0
    HISTORICAL_DATA_WEIGHT:          float = 0.3
    BLOCK_TIME:      int   = 12
    BLOCKS_PER_HOUR: int   = 300
    TOKEN_PRICE:     float = 2000.0
    API_HOST:             str   = "0.0.0.0"
    API_PORT:             int   = 8000
    API_RATE_LIMIT_DELAY: float = 0.25

settings = Settings()
