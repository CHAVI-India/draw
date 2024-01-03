import datetime

from sqlalchemy import String, Column, Enum, DateTime, BigInteger
from draw.dao.common import Base, Model, Status


class DicomLog(Base):
    __tablename__ = "dicomlog"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    series_name = Column("series_name", String(256))
    input_path = Column("input_path", String(1024))
    output_path = Column("output_path", String(1024))
    status = Column(
        "status",
        Enum(Status),
        server_default=str(Status.INIT.name),
        index=True,
    )
    model = Column("model", Enum(Model))
    created_on = Column(
        "created_on",
        DateTime,
        server_default=str(datetime.datetime.utcnow()),
    )

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
