from typing import List

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import text

from .table import DicomLog
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
        if not self.engine.dialect.has_table(
            self.engine.connect(), self._table_name
        ):
            DicomLog.metadata.create_all(self.engine)
            print("Created Table")

    def top(self, dataset_name):
        if dataset_name not in MODEL_CONFIG["KEYS"]:
            raise KeyError(f"{dataset_name} not in {MODEL_CONFIG['KEYS']}")
        query = (
            "SELECT * "
            "FROM {} "
            "WHERE model='{}' "
            "AND status='INIT' "
            "ORDER BY created_on DESC "
            "LIMIT {}"
        ).format(self._table_name, dataset_name, self._batch_size)
        print(query)
        with self.engine.connect() as conn:
            result_set = conn.execute(text(query))

            for result in result_set:
                print(result)

    def insert(self, dcm_log: List[DicomLog]):
        self.session.add_all(dcm_log)
        self.session.commit()

    def update(
        self,
    ):
        pass
