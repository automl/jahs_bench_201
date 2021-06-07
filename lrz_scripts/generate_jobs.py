#!/usr/bin/python

import sys
from pathlib import Path
import argparse
from string import Template

parser = argparse.ArgumentParser(description="Generate SLURM job files using a template.")
parser.add_argument("--partition", type=str, default="test", help="Intended partition for the job. Default: test", 
                       choices = ["test", "micro", "general", "large"])
parser.add_argument("--nnodes", type=int, default=1, help="Number of nodes to be used by the job.")
parser.add_argument("--template", type=Path, help="A template file which will be used to generate the job file.")
parser.add_argument("--output", type=Path, help="Path to the final job file.")
args = parser.parse_args()

with open(args.template.expanduser().absolute(), "r") as fp:
    template = Template(fp.read())

output_file: Path = args.output.absolute()

if output_file.exists() and not output_file.is_file():
    raise TypeError(f"The given output path does not refer to a file, cannot overwrite.")

assert args.nnodes >= 1, f"The number of nodes requested by a job must be a positive integer, was {args.nnodes}."
output = template.substitute(partition=args.partition, nnodes=args.nnodes)

with open(output_file, "w") as fp:
    fp.write(output)
