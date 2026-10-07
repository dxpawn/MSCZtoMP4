# SPDX-License-Identifier: AGPL-3.0-or-later
import sys

from .cli import main

if __name__ == '__main__':     # required: render workers re-import the main module (spawn start method)
    sys.exit(main())
