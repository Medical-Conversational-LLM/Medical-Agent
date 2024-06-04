from reader.reader import Reader
from typing import Generator, Any, Callable
 
class MemoryReader(Reader):
 
    data: list[dict[str,str]] 

    def __init__(
        self, data: list[dict[str,str]] 
    ) -> None:
        super().__init__() 
        self.data = data  
    def read(self) -> Generator[int, None, dict[str, Any]]:
        for line in self.data:
            yield line

