"""Database connection shared by all ETL scripts.

The connection string comes from (first match wins):
  1. the DATABASE_URL environment variable
  2. DATABASE_URL in the repo's .env file
  3. the docker-compose default (localhost:5432)
"""
import os
import pathlib

import psycopg

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_URL = "postgresql://windcast:windcast_dev@localhost:5432/windcast"


def database_url() -> str:
    if os.environ.get("DATABASE_URL"):
        return os.environ["DATABASE_URL"]
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("DATABASE_URL="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return DEFAULT_URL


def connect() -> psycopg.Connection:
    return psycopg.connect(database_url())
