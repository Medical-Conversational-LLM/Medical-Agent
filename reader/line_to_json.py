from reader.reader import Reader
from typing import Generator, Any, Callable
import json

class LineToJson(Reader):
 
    file: str 

    def __init__(
        self, file: str
    ) -> None:
        super().__init__() 
        self.file = file 
        self.id = file

    def read(self) -> Generator[int, None, dict[str, Any]]:
        with open(self.file, "r") as file:
            for line in file:          
                yield  json.loads(line.strip())
