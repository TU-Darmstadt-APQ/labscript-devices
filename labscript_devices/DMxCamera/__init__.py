
from labscript_utils import check_version

import sys
if sys.version_info < (3, 6):
    raise RuntimeError("DMx camera requires Python 3.6+")
