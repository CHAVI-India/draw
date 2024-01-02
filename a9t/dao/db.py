from typing import List, Any

from sqlalchemy import Insert, create_engine
from sqlalchemy.sql import text

from a9t.dao.table import DicomLog, Status
from a9t.config import DB_CONFIG, MODEL_CONFIG, PRED_BATCH_SIZE, LOG

_ENGINE = create_engine(
    DB_CONFIG["URL"],
    echo=False,
    isolation_level="READ UNCOMMITTED",
    pool_size=10,
    max_overflow=20,
    pool_timeout=100,
)


class DBConnection:
    """
    Connection to DB
    """

    def __init__(self):
        self.engine = _ENGINE
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
            "ORDER BY created_on ASC "
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
        try:
            with self.engine.connect() as conn:
                conn.execute(Insert(DicomLog), [d.get_attr_dict() for d in dcm_log])
                conn.commit()
                LOG.info(f"ENQUEUED {[d.input_path for d in dcm_log]}")

        except Exception:
            LOG.error(f"Exception for {dcm_log}", exc_info=True)

    def update_status(
        self,
        dcm_log: DicomLog,
        updated_status: Status,
    ):
        if dcm_log.id is None:
            LOG.error(f"NO ID to update")
            return

        q = "UPDATE dicomlog SET status='{}' WHERE dicomlog.id='{}'".format(
            updated_status.value,
            dcm_log.id,
        )
        with self.engine.connect() as conn:
            conn.execute(text(q))
            conn.commit()
        LOG.info(f"Updated Status of {dcm_log.id} to {updated_status.value}")

    @staticmethod
    def convert_to_obj(row: Any) -> DicomLog:
        obj = DicomLog(
            id=row.id,
            series_name=row.series_name,
            input_path=row.input_path,
            output_path=row.output_path,
            status=row.status,
            model=row.model,
            created_on=row.created_on,
        )
        return obj

    def update_status(self, series_name: str, output_path: str, status: Status):
        q = "UPDATE dicomlog SET status='{}', output_path='{}' WHERE dicomlog.series_name='{}'".format(
            status,
            series_name,
            output_path,
        )
        with self.engine.connect() as conn:
            conn.execute(text(q))
            conn.commit()
        LOG.info(f"Updated Status of {series_name} to {status}")
