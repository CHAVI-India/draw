from typing import List

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import text

from .table import DicomLog, Status
from ..config import DB_CONFIG, MODEL_CONFIG


class DBConnection:
    """
    Connection to DB
    """

    def __init__(self):
        self.engine = create_engine(DB_CONFIG["URL"], echo=True)
        self.session = sessionmaker(bind=self.engine)()
        self._table_name = DicomLog.__tablename__
        self._batch_size = 4
        self.create_table_if_not_exists()

    def create_table_if_not_exists(self):
        if not self.engine.dialect.has_table(self.engine.connect(), self._table_name):
            DicomLog.metadata.create_all(self.engine)
            print("Created Table")

    def top(self, dataset_name, status: Status):
        if dataset_name not in MODEL_CONFIG["KEYS"]:
            raise KeyError(f"{dataset_name} not in {MODEL_CONFIG['KEYS']}")
        query = (
            "SELECT * "
            "FROM {} "
            "WHERE model='{}' "
            "AND status='{}' "
            "ORDER BY created_on DESC "
            "LIMIT {}"
        ).format(self._table_name, dataset_name, status.value, self._batch_size)
        print(query)
        with self.engine.connect() as conn:
            result_set = conn.execute(text(query))
            return [self.convert_to_obj(row) for row in result_set]

    def insert(self, dcm_log: List[DicomLog]):
        self.session.add_all(dcm_log)
        self.session.commit()

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

    @staticmethod
    def convert_to_obj(row):
        obj = DicomLog(
            id=row.id,
            input_path=row.input_path,
            output_path=row.output_path,
            status=row.status,
            model=row.model,
            created_on=row.created_on,
        )
        return obj

    def update(self, series_name, output_path):
        q = "UPDATE dicomlog SET status='{}', output_path='{}' WHERE dicomlog.series_name='{}'".format(
            Status.PREDICTED.value,
            series_name,
            output_path,
        )
        with self.engine.connect() as conn:
            conn.execute(text(q))
            conn.commit()
