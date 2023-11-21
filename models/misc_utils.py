import click
import warnings

# We need to import this as early as possible. This
# is a hack to get around the following issue.
# https://github.com/pytorch/pytorch/issues/109442
from torch._inductor.codecache import AsyncCompile


def swarn(msg: str, **kwargs) -> None:
    styled_msg = click.style(msg, **kwargs)
    warnings.warn(styled_msg)
