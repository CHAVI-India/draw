from a9t.dao.db import DBConnection
from a9t.dao.table import DicomLog

conn = DBConnection()

if __name__ == "__main__":
    # all_dcm = [
    #     DicomLog(input_path="-1.1test1", output_path="test2", model="TSPrime"),
    #     DicomLog(input_path="-1.2test1.1", output_path="test2", model="TSPrime"),
    #     DicomLog(input_path="-1.3test1.2", output_path="test2", model="TSPrime"),
    #     DicomLog(input_path="-1.4test1.3", output_path="test2", model="TSPrime"),
    # ]
    # conn.insert(
    #     all_dcm
    # )
    conn.top("TSPrime")
