"""
YAML config support for all scripts in this repo.

Usage in a script:

    import argparse
    from utils_config import parse_args

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    ...
    args = parse_args(parser)   # instead of parser.parse_args()

Precedence (lowest -> highest):
    argparse defaults  <  YAML file (--config)  <  explicit CLI flags

YAML keys may use either hyphens or underscores (``data-path`` / ``data_path``).
Unknown keys in the YAML raise an error so typos do not silently fall back to
defaults.
"""
import argparse
import sys

import yaml


def _load_yaml(path):
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: top level must be a mapping, got {type(cfg).__name__}")
    return {k.replace("-", "_"): v for k, v in cfg.items()}


def parse_args(parser, argv=None):
    """Parse CLI args with optional ``--config file.yaml`` providing defaults."""
    if argv is None:
        argv = sys.argv[1:]
    parser.add_argument("--config", type=str, default=None,
                        help="YAML file with default values; explicit CLI flags override it")

    # Pre-scan with a bare parser: the real one would reject missing
    # required args before the YAML gets a chance to supply them.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None)
    known, _ = pre.parse_known_args(argv)
    if known.config:
        cfg = _load_yaml(known.config)
        valid = {a.dest for a in parser._actions}
        unknown = sorted(set(cfg) - valid)
        if unknown:
            parser.error(f"unknown keys in {known.config}: {', '.join(unknown)} "
                         f"(valid: {', '.join(sorted(valid - {'help', 'config'}))})")
        # YAML satisfies required args and overrides defaults; CLI still wins.
        for action in parser._actions:
            if action.dest in cfg:
                action.required = False
                if action.choices is not None and cfg[action.dest] not in action.choices:
                    parser.error(f"{known.config}: {action.dest}={cfg[action.dest]!r} "
                                 f"not in {list(action.choices)}")
        parser.set_defaults(**cfg)

    return parser.parse_args(argv)
