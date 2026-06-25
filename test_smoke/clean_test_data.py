'''Clean transient artifacts under test_data/.

LIST_PATH_CLEAN is empty: the smoke + data-dependent tests emit no persistent
output under test_data/. If a future test writes persistent evaluation output
(prediction dumps, metric tables), register its generated subdir here
(convention: test_data/_tmp_<topic>/).

User-supplied sample data lives directly under test_data/ and must NEVER be
touched by this script; only generated subdirs belong in LIST_PATH_CLEAN.
'''

import shutil
from pathlib import Path
from typing import List


PATH_REPO_ROOT = Path(__file__).resolve().parent.parent
LIST_PATH_CLEAN: List[Path] = []


def main() -> None:
    if len(LIST_PATH_CLEAN) == 0:
        print('nothing configured to clean; edit LIST_PATH_CLEAN in %s' % __file__)
        return

    for path in LIST_PATH_CLEAN:
        if path.exists():
            shutil.rmtree(path)
            print('removed: %s' % path)
        else:
            print('skip (does not exist): %s' % path)


if __name__ == '__main__':
    main()
