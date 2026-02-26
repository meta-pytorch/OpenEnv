# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Generic agent runner entry point.

This module is invoked when running ``python -m agentic.kernel.core.runner``.
It is not a standalone entry point — use an agent-type-specific runner
(set via each plugin's resolve_command()).
"""

import sys


def main() -> None:
    print(
        "Error: agentic.kernel.core.runner is not a standalone entry point.\n"
        "Use an agent-type-specific runner instead.\n"
        "Each AgentTypePlugin provides its own runner via resolve_command().",
        file=sys.stderr,
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
