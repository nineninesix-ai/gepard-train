#!/usr/bin/env python3
"""Dataset-preparation entry point — thin alias for `gepard.data.preprocessing.prepare`.

    python -m gepard.cli.prepare [--output PATH] [--n-shards N] [overrides...]
"""

from gepard.data.preprocessing.prepare import main

if __name__ == "__main__":
    main()
