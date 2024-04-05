from datetime import datetime
from sqlalchemy import String, Column, Enum, DateTime, BigInteger
from draw.dao.common import Base, Model, Status
from sqlalchemy.sql import func
from sqlalchemy.dialects import postgresql, mysql, sqlite


BigIntegerType = BigInteger()
BigIntegerType = BigIntegerType.with_variant(postgresql.BIGINT(), "postgresql")
BigIntegerType = BigIntegerType.with_variant(mysql.BIGINT(), "mysql")
BigIntegerType = BigIntegerType.with_variant(sqlite.INTEGER(), "sqlite")


class DicomLog(Base):
    __tablename__ = "dicomlog"
    id = Column(BigIntegerType, primary_key=True, autoincrement=True)
    series_name = Column("series_name", String(256), nullable=False, unique=True)
    input_path = Column("input_path", String(1024), nullable=False)
    output_path = Column("output_path", String(1024), nullable=True)
    status = Column(
        "status",
        Enum(Status),
        default=Status.INIT.value,
        server_default=str(Status.INIT.value),
        index=True,
        nullable=False,
    )
    model = Column("model", Enum(Model), nullable=False)
    created_on = Column(
        "created_on",
        DateTime,
        default=datetime.now,
        server_default=func.current_timestamp(),
        nullable=False,
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

    def set_status(self, status: Status):
        self.status = status
