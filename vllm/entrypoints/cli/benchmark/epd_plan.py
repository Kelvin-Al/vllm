# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse

from vllm.benchmarks.epd_plan import add_cli_args, main
from vllm.entrypoints.cli.benchmark.base import BenchmarkSubcommandBase


class BenchmarkEPDPlanSubcommand(BenchmarkSubcommandBase):
    """Estimate an E+PD GPU split from two calibrated request rates."""

    name = "epd-plan"
    help = "Plan GPU allocation for disaggregated encoding with combined PD."

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        add_cli_args(parser)

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        main(args)
