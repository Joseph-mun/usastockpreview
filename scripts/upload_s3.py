# -*- coding: utf-8 -*-
"""Upload prediction signal to S3."""

import sys

import boto3

from src.config import DATA_DIR, get_logger

logger = get_logger(__name__)

S3_BUCKET = "alloc-trading-signals"
S3_KEY = "prediction_signal.json"
S3_REGION = "ap-northeast-2"


def main():
    signal_path = DATA_DIR / "prediction_signal.json"
    if not signal_path.exists():
        logger.warning("prediction_signal.json not found, skipping S3 upload")
        sys.exit(0)

    s3 = boto3.client("s3", region_name=S3_REGION)
    s3.upload_file(
        str(signal_path),
        S3_BUCKET,
        S3_KEY,
        ExtraArgs={"ContentType": "application/json"},
    )
    logger.info("Uploaded %s to s3://%s/%s", signal_path, S3_BUCKET, S3_KEY)

    # Upload probability history for dashboard
    history_path = DATA_DIR / "probability_history.csv"
    if history_path.exists():
        s3.upload_file(
            str(history_path),
            S3_BUCKET,
            "probability_history.csv",
            ExtraArgs={"ContentType": "text/csv"},
        )
        logger.info("Uploaded %s to s3://%s/probability_history.csv", history_path, S3_BUCKET)


if __name__ == "__main__":
    main()
