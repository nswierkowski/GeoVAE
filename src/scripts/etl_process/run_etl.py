import sys
import os
from src.scripts.etl_process.ETLProcessor import ETLProcessor

if __name__ == "__main__":
    if len(sys.argv) > 1:
        DATASET_ID = sys.argv[1]
    else:
        DATASET_ID = "mahmudulhaqueshawon/cat-image" 

    dataset_name = DATASET_ID.replace("/", "_").lower()
    DATA_DIR = os.path.join("data", "raw_data", dataset_name)
    SPLIT_DATA_DIR = os.path.join("data", "data_splits", dataset_name)

    etl_processor: ETLProcessor = ETLProcessor(DATASET_ID, DATA_DIR, SPLIT_DATA_DIR)

    print(f"Starting ETL process for dataset: {DATASET_ID}")
    etl_processor.process()
    print("ETL process completed successfully.")
