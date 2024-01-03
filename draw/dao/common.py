import enum
from sqlalchemy import create_engine

from draw.config import DB_CONFIG, MODEL_CONFIG
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


class Status(enum.Enum):
    INIT = "INIT"
    STARTED = "STARTED"
    PREDICTED = "PREDICTED"
    SENT = "SENT"


Model = enum.Enum("Model", tuple(MODEL_CONFIG["KEYS"]))
