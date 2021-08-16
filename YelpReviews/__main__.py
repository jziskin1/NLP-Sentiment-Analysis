#!/usr/bin/env python

import sys
from .cli import main as main_cli

def main(args):
    main_cli(args)

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))