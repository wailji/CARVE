"""Subprocess-isolated execution sandbox for assert-list grading (MBPP-style).

Wraps `human_eval.execution.unsafe_execute` semantics — runs `program` in a
fresh process with reliability_guard, returns "passed" / "failed: …" /
"timed out".

Why not human_eval.check_correctness directly? It expects a `problem` dict
with HumanEval-specific keys (`prompt`, `entry_point`, `test`) and constructs
the program as `prompt + completion + test + check(entry_point)`. MBPP
grading has a different shape (test_imports + completion + assert list, no
`check()` call). Easier to mirror unsafe_execute with our own program string.
"""
from __future__ import annotations

from multiprocessing import Manager, Process
from typing import Dict


def _unsafe_run(program: str, timeout: float, result) -> None:  # type: ignore[no-untyped-def]
    # Mirror human_eval.execution.unsafe_execute: tempdir + reliability_guard.
    from human_eval.execution import (
        create_tempdir,
        reliability_guard,
        swallow_io,
        time_limit,
        TimeoutException,
    )
    import os as _os
    import shutil as _shutil

    with create_tempdir():
        rmtree = _shutil.rmtree
        rmdir = _os.rmdir
        chdir = _os.chdir
        reliability_guard()
        try:
            with swallow_io():
                with time_limit(timeout):
                    exec(program, {})
            result.append("passed")
        except TimeoutException:
            result.append("timed out")
        except BaseException as e:  # noqa: BLE001
            result.append(f"failed: {e}")
        # restore for tempdir cleanup
        _shutil.rmtree = rmtree
        _os.rmdir = rmdir
        _os.chdir = chdir


def run_program(program: str, timeout: float = 5.0) -> Dict[str, object]:
    """Execute `program` in a sandboxed subprocess, returns {passed, result}."""
    with Manager() as mgr:
        result = mgr.list()
        p = Process(target=_unsafe_run, args=(program, timeout, result))
        p.start()
        p.join(timeout=timeout + 1)
        if p.is_alive():
            p.kill()
        if not result:
            result.append("timed out")
        return {"passed": result[0] == "passed", "result": str(result[0])}
