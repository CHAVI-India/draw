from subprocess import Popen

if __name__ == "__main__":
    commands = ["python ./TASK_copy.py", "python ./TASK_predict.py"]

    procs = [Popen(i) for i in commands]

    for p in procs:
        p.wait()
