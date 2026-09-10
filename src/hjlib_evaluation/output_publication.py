'''Failure-clean staged publication for one new output directory.'''
from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
import shutil
import tempfile

from hjlib_systools.fileops import commit_directory_no_replace


@contextmanager
def staged_output_directory(
    path_output_root: Path,
) -> Generator[Path, None, None]:
    '''Yield a sibling staging root and publish it only after successful exit.'''
    final = path_output_root.absolute()
    if final.exists():
        raise FileExistsError('output root already exists: %s' % final)
    final.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(
        prefix='.%s.partial-' % final.name,
        dir=final.parent,
    ))
    try:
        yield stage
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    commit_directory_no_replace(stage, final)


__all__ = ['staged_output_directory']
