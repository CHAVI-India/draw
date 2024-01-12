from typing import List


from draw.dao.table import DicomLog
from draw.config import PRED_BATCH_SIZE, LOG
from draw.dao.common import DB_ENGINE, Status
from sqlalchemy.orm import Session
from sqlalchemy import exists, select, update


def dequeue_from_db(dcm: DicomLog):
    DBConnection.update_status_by_id(dcm, Status.STARTED)
    dcm.set_status(Status.STARTED)
    return dcm


class DBConnection:
    """Querying the database and interacting with the records"""

    @staticmethod
    def dequeue(model: str) -> List[DicomLog]:
        """Dequeue batched records from the DB

        Args:
            model (str): Model name which record to fetch

        Returns:
            List[DicomLog]: Batch of DICOM records to run prediction on
        """
        try:
            top_dcm_logs = [
                dequeue_from_db(dcm) for dcm in DBConnection.top(model, Status.INIT)
            ]
            LOG.info(f"Dequeing {len(top_dcm_logs)}")
            return top_dcm_logs
        except Exception:
            LOG.error("ERROR while dequeing", exc_info=True)
            return []

    @staticmethod
    def exists(series_name: str) -> bool:
        """Check if Series Name exists in DB or not

        Args:
            series_name (str): Series Instance UID of the DICOM file

        Returns:
            bool: If the series_name exists in the DB or not
        """
        with Session(DB_ENGINE) as sess:
            exists_query = sess.query(
                exists().where(DicomLog.series_name == series_name)
            )
            return sess.execute(exists_query).scalar()

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
                    .order_by(DicomLog.created_on)
                    .limit(PRED_BATCH_SIZE)
                )
                return sess.scalars(stmt).all()
        except:
            LOG.error(f"Error while Fetching TOP {status} {model}", exc_info=True)
            return []

    @staticmethod
    def enqueue(records: List[DicomLog]):
        """Inserts list of records into the DB

        Args:
            records (List[DicomLog]): Records to insert in the DB
        """
        try:
            lr = len(records)
            with Session(DB_ENGINE) as sess:
                sess.add_all(records)
                sess.commit()
            LOG.info(f"Enqued {lr}")
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
