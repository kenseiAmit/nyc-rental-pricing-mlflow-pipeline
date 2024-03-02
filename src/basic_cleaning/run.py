#!/usr/bin/env python
"""Download a W&B artifact, perform basic cleaning, and log the result."""

import argparse
import logging
import pandas as pd
import wandb


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    logger.info("Downloading input artifact")
    artifact_local_path = run.use_artifact(args.input_artifact).file()

    logger.info("Loading data into dataframe")
    df = pd.read_csv(artifact_local_path)

    logger.info(f"Filtering price between {args.min_price} and {args.max_price}")
    idx = df["price"].between(args.min_price, args.max_price)
    df = df[idx].copy()

    logger.info("Converting last_review to datetime")
    df["last_review"] = pd.to_datetime(df["last_review"])

    logger.info("Filtering proper longitude and latitude boundaries for NYC")
    geo_idx = df["longitude"].between(-74.25, -73.50) & df["latitude"].between(40.5, 41.2)
    df = df[geo_idx].copy()

    logger.info("Saving cleaned data and logging artifact")
    clean_path = "clean_sample.csv"
    df.to_csv(clean_path, index=False)

    artifact = wandb.Artifact(
        name=args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(clean_path)
    run.log_artifact(artifact)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform basic data cleaning")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Input artifact name in W&B",
        required=True,
    )
    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Output artifact name in W&B",
        required=True,
    )
    parser.add_argument(
        "--output_type",
        type=str,
        help="Output artifact type",
        required=True,
    )
    parser.add_argument(
        "--output_description",
        type=str,
        help="Output artifact description",
        required=True,
    )
    parser.add_argument(
        "--min_price",
        type=float,
        help="Minimum allowed price (inclusive)",
        required=True,
    )
    parser.add_argument(
        "--max_price",
        type=float,
        help="Maximum allowed price (inclusive)",
        required=True,
    )

    args = parser.parse_args()
    go(args)
