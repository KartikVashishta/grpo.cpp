"""Drive the native model with token IDs and rewards."""

import json
import os
from pathlib import Path
import select
import subprocess
import time


class Native:
    def __init__(self, model, log, learning_rate=1e-6, memcheck=False):
        command = ["/build/qwen_train", str(model), str(learning_rate)]
        if memcheck:
            command = ["compute-sanitizer", "--tool", "memcheck", "--error-exitcode", "9"] + command
        self.log = Path(log).open("w")
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self.log, bufsize=0)
        self.peak_used = 0
        try:
            if self.read() != {"ready": True}:
                raise RuntimeError("Native model did not start")
        except BaseException:
            self.close()
            raise

    def read(self):
        deadline = time.monotonic() + 300
        result = bytearray()
        while not result.endswith(b"\n"):
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([self.process.stdout], [], [], remaining)[0]:
                raise TimeoutError("Native model took more than five minutes to reply")
            chunk = os.read(self.process.stdout.fileno(), 65536)
            if not chunk:
                raise RuntimeError(f"Native model closed its output; see {self.log.name}")
            result.extend(chunk)
        response = json.loads(result)
        if "error" in response:
            raise RuntimeError(response["error"])
        self.peak_used = max(self.peak_used, response.get("peak_device_used_bytes", 0))
        return response

    def call(self, command, **arguments):
        payload = (json.dumps({"command": command, **arguments}, allow_nan=False) + "\n").encode()
        pending = memoryview(payload)
        while pending:
            pending = pending[os.write(self.process.stdin.fileno(), pending):]
        return self.read()

    def close(self):
        self.process.stdin.close()
        try:
            code = self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
            code = -9
        self.process.stdout.close()
        self.log.close()
        return code

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        code = self.close()
        if kind is None and code:
            raise RuntimeError(f"Native model exited with code {code}; see {self.log.name}")


def prompt_ids(tokenizer, puzzles):
    from task import messages
    return [tokenizer.apply_chat_template(messages(p), tokenize=True, add_generation_prompt=True,
                                          enable_thinking=False) for p in puzzles]


def eos_ids(model):
    values = json.loads((Path(model) / "generation_config.json").read_text())["eos_token_id"]
    return [values] if isinstance(values, int) else values
