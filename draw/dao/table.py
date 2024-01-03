import datetime
import enum

from draw.config import MODEL_CONFIG
from sqlalchemy import String, Column, Enum, DateTime, BigInteger
from draw.dao.common import Base

Model = enum.Enum("Model", tuple(MODEL_CONFIG["KEYS"]))


class Status(enum.Enum):
    INIT = "INIT"
    STARTED = "STARTED"
    PREDICTED = "PREDICTED"
    SENT = "SENT"


class DicomLog(Base):
    __tablename__ = "dicomlog"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    series_name = Column("series_name", String(256))
    input_path = Column("input_path", String(2000))
    output_path = Column("output_path", String(2000))
    status = Column("status", Enum(Status), default=Status.INIT, index=True)
    model = Column("model", Enum(Model))
    created_on = Column("created_on", DateTime, default=datetime.datetime.utcnow)

    def __repr__(self):
        return (
            f"DicomLog("
            f"id={self.id}, "
            f"series_name={self.series_name}, "
            f"input_path={self.input_path}, "
            f"output_path={self.output_path}, "
            f"status={self.status}, "
            f"model={self.model}, "
            f"created_on={self.created_on})"
        )

    def get_attr_dict(self):
        return {
            "id": self.id,
            "series_name": self.series_name,
            "input_path": self.input_path,
            "output_path": self.output_path,
            "status": self.status,
            "model": self.model,
            "created_on": self.created_on,
        }
