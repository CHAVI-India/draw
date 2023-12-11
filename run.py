from subprocess import Popen

if __name__ == "__main__":
    commands = ["python a9t/pipeline/TASK_copy.py", "python a9t/pipeline/TASK_predict.py"]

    procs = [Popen(i.split()) for i in commands]

    for p in procs:
        p.wait()
