from typing import List


from draw.dao.table import DicomLog
from draw.config import PRED_BATCH_SIZE, LOG
from draw.dao.common import DB_ENGINE, Status
from sqlalchemy.orm import Session
from sqlalchemy import select, update


class DBConnection:
    """Querying the database and interacting with the records"""

    @staticmethod
    def top(model: str, status: Status) -> List[DicomLog]:
        """Gives List of Data to process from DB

        Args:
            dataset_name (str): model name like TSPrime, TSGyne etc
            status (Status): Status to filter the records

        Returns:
            List[DicomLog]: batch list of records that match the criteria
        """
        try:
            with Session(DB_ENGINE) as sess:
                stmt = (
                    select(DicomLog)
                    .where(DicomLog.model == model)
                    .where(DicomLog.status == status)
                    .order_by(DicomLog.created_on.desc())
                    .limit(PRED_BATCH_SIZE)
                )
                return sess.scalars(stmt).all()
        except:
            LOG.error(f"Error while Fetching TOP {status} {model}", exc_info=True)
            return []

    @staticmethod
    def insert(records: List[DicomLog]):
        """Inserts list of records into the DB

        Args:
            records (List[DicomLog]): Records to insert in the DB
        """
        try:
            with Session(DB_ENGINE) as sess:
                sess.add_all(records)
                sess.commit()
        except:
            LOG.error(f"Could not insert Records", exc_info=True)

    @staticmethod
    def update_status_by_id(dcm_log: DicomLog, updated_status: Status):
        """Updates Status of Given Log

        Args:
            dcm_log (DicomLog): Log Record
            status (Status): Status to Update
        """
        with Session(DB_ENGINE) as sess:
            stmt = (
                update(DicomLog)
                .where(DicomLog.id == dcm_log.id)
                .values(status=updated_status)
            )
            sess.execute(stmt)
            sess.commit()

    @staticmethod
    def update_record_by_series_name(
        series_name: str, output_path: str, status: Status
    ):
        """Updates Status by Series name

        Args:
            series_name (str): Series Name of the record to update
            output_path (str): output path of the record
            status (Status): updated status
        """
        with Session(DB_ENGINE) as sess:
            stmt = (
                update(DicomLog)
                .where(DicomLog.series_name == series_name)
                .values(status=status)
                .values(output_path=output_path)
            )
            sess.execute(stmt)
            sess.commit()
