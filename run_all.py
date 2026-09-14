"""Run the complete analysis: tables, robustness analyses, figures and the
reproduction check.

    python run_all.py

Requires the two SPSS files in the directory given by the environment variable
ELASTIBD_DATA (default ./data). Nothing is written outside ``outputs/``.
"""

from __future__ import annotations

import sys

import analysis_supplementary
import analysis_tables
import figures
import robustness
import verify


def main() -> int:
    print("== main tables ==")
    analysis_tables.main()
    print("\n== supplementary tables ==")
    analysis_supplementary.main()
    print("\n== robustness analyses ==")
    robustness.main()
    print("\n== figures ==")
    figures.main()
    print("\n== reproduction check ==")
    return verify.main()


if __name__ == "__main__":
    sys.exit(main())
