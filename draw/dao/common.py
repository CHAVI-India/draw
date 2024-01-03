from sqlalchemy import create_engine

from draw.config import DB_CONFIG
from sqlalchemy.ext.declarative import declarative_base

DB_ENGINE = create_engine(
    DB_CONFIG["URL"],
    echo=False,
    isolation_level="READ UNCOMMITTED",
    pool_size=10,
    max_overflow=20,
    pool_timeout=100,
)

Base = declarative_base()
