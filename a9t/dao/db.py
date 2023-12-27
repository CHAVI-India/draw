from typing import List, Any

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import text

from .table import DicomLog, Status
from ..config import DB_CONFIG, MODEL_CONFIG, PRED_BATCH_SIZE, LOG

_ENGINE = create_engine(
    DB_CONFIG["URL"],
    echo=False,
    isolation_level="READ UNCOMMITTED",
    pool_size=10,
    max_overflow=20,
    pool_timeout=100,
)

_SESSION = sessionmaker(bind=_ENGINE)()


class DBConnection:
    """
    Connection to DB
    """

    def __init__(self):
        self.engine = _ENGINE
        self.session = _SESSION
        self._table_name = DicomLog.__tablename__
        self.BATCH_SIZE = PRED_BATCH_SIZE
        self.create_table_if_not_exists()

    def create_table_if_not_exists(self):
        """Creates Default table if not exists"""
        with self.engine.connect() as conn:
            if not self.engine.dialect.has_table(conn, self._table_name):
                DicomLog.metadata.create_all(self.engine)
                LOG.info(f"Table {self._table_name} created")

    def top(self, dataset_name: str, status: Status) -> List[DicomLog]:
        if dataset_name not in MODEL_CONFIG["KEYS"]:
            raise KeyError(f"{dataset_name} not in {MODEL_CONFIG['KEYS']}")
        query = (
            "SELECT * "
            "FROM {} "
            "WHERE model='{}' "
            "AND status='{}' "
            "ORDER BY created_on DESC "
            "LIMIT {}"
        ).format(self._table_name, dataset_name, status.value, self.BATCH_SIZE)
        try:
            with self.engine.connect() as conn:
                result_set = conn.execute(text(query))
                result_objects = [self.convert_to_obj(row) for row in result_set]
                LOG.info(f"Found {len(result_objects)}")
                return result_objects
        except Exception:
            return []

    def insert(self, dcm_log: List[DicomLog]):
        self.session.add_all(dcm_log)
        self.session.commit()
        LOG.info(f"Added {len(dcm_log)} to DB")

    def update_status(
        self,
        dcm_log: DicomLog,
        updated_status: Status,
    ):
        q = "UPDATE dicomlog SET status='{}' WHERE dicomlog.id='{}'".format(
            updated_status.value, dcm_log.id
        )
        with self.engine.connect() as conn:
            conn.execute(text(q))
            conn.commit()
        LOG.info(f"Updated Status of {dcm_log.id} to {updated_status.value}")

    def update_status_with_series_name(
        self,
        dcm_series_name: str,
        updated_status: Status,
    ):
        q = "UPDATE dicomlog SET status='{}' WHERE dicomlog.id='{}'".format(
            updated_status.value, dcm_series_name
        )
        with self.engine.connect() as conn:
            conn.execute(text(q))
            conn.commit()
        LOG.info(f"Updated Status of Series {dcm_series_name} to {updated_status.value}")

    @staticmethod
    def convert_to_obj(row: Any) -> DicomLog:
        obj = DicomLog(
            id=row.id,
            input_path=row.input_path,
            output_path=row.output_path,
            status=row.status,
            model=row.model,
            created_on=row.created_on,
        )
        return obj

    def update(self, series_name: str, output_path: str):
        """Updates Status and Output Path of given series name"""
        q = "UPDATE dicomlog SET status='{}', output_path='{}' WHERE dicomlog.series_name='{}'".format(
            Status.PREDICTED.value,
            series_name,
            output_path,
        )
        with self.engine.connect() as conn:
            conn.execute(text(q))
            conn.commit()
        LOG.info(f"Updated Status of {series_name} to {Status.PREDICTED.value}")
