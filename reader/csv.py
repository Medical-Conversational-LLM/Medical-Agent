from reader.reader import Reader
import csv
from typing import Generator, Any, Callable


class CSV(Reader):

    skip_header = False
    file: str
    mapper: Callable[[str, str], str]

    def __init__(
        self, file: str, skip_header=False, mapper: Callable[[list[str]], str] = None
    ) -> None:
        super().__init__()
        self.skip_header = skip_header
        self.file = file
        self.mapper = mapper
        self.id = file

    def read(self) -> Generator[int, None, dict[str, Any]]:
        with open(self.file, "r") as file:
            csvreader = csv.reader(file)
            if self.skip_header:
                next(csvreader)

            for row in csvreader: 
                if self.mapper is not None:
                    yield self.mapper(row)
                else:
                    yield row
