import gzip
import os
import shutil
from pathlib import Path
from typing import Iterable


def gzip_csv_files(paths: Iterable[str | Path]) -> list[Path]:
    """Atomically gzip CmdStan CSVs while retaining files CmdStanPy may reread."""
    preserved = []
    for path_value in paths:
        source = Path(path_value)
        destination = source.with_suffix(source.suffix + ".gz")
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        with source.open("rb") as input_file, gzip.open(temporary, "wb") as output_file:
            shutil.copyfileobj(input_file, output_file)
        os.replace(temporary, destination)
        preserved.append(destination)
    return preserved