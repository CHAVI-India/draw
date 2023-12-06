import datetime
import enum
import uuid

from sqlalchemy import String, Column, Enum, Uuid, DateTime
from sqlalchemy.orm import DeclarativeBase

from ..mapping import ALL_SEG_MAP

Model = enum.Enum("Model", tuple(ALL_SEG_MAP.keys()))


class Status(enum.Enum):
    INIT = "INIT"
    STARTED = "STARTED"
    COMPLETED = "COMPLETED"
    PROCESSING = "PROCESSING"


class Base(DeclarativeBase):
    pass


class DicomLog(Base):
    __tablename__ = "dicomlog"
    id = Column(Uuid, primary_key=True, default=uuid.uuid4)
    input_path = Column("input_path", String(2000))
    output_path = Column("output_path", String(2000))
    status = Column("status", Enum(Status), default=Status.INIT)
    model = Column("model", Enum(Model))
    created_on = Column("created_on", DateTime, default=datetime.datetime.utcnow)

    def __repr__(self):
        return (
            f"DicomLog("
            f"id={self.id}, "
            f"input_path={self.input_path}, "
            f"output_path={self.output_path}, "
            f"status={self.status}, "
            f"model={self.model}, "
            f"created_on={self.created_on})"
        )
