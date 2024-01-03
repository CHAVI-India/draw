from typing import List, Any

from sqlalchemy import Insert
from sqlalchemy.sql import text

from draw.dao.table import DicomLog
from draw.config import MODEL_CONFIG, PRED_BATCH_SIZE, LOG
from draw.dao.common import DB_ENGINE, Status


class DBConnection:
    """
    Connection to DB
    """

    def __init__(self):
        self.engine = DB_ENGINE
        self._table_name = DicomLog.__tablename__
        self.BATCH_SIZE = PRED_BATCH_SIZE

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
