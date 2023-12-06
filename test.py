from a9t.dao.db import DBConnection
from a9t.dao.table import DicomLog, Status

conn = DBConnection()

if __name__ == "__main__":
    all_dcm = conn.top("TSPrime")
    print("init", *all_dcm, sep="\n")
    conn.update_status(all_dcm[0], Status.COMPLETED)
    print(*conn.top("TSPrime"), sep="\n")
