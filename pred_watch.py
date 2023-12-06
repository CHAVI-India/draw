import subprocess as sp

if __name__ == "__main__":
    memory_used_command = ["nvidia-smi", "--query-gpu=memory.used", "--format=csv"]
    isolate_memory_value = lambda x: "".join(
        y for y in x.decode("ascii") if y in "0123456789"
    )
    memory_used = isolate_memory_value(
        sp.check_output(memory_used_command, stderr=sp.STDOUT)
    )

    print(memory_used)
