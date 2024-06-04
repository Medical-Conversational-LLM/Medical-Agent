from writer.writer import Writer
import json


class JsonWriter(Writer):
    output_file: str

    def __init__(self, output_file: str) -> None:
        super().__init__()
        self.output_file = output_file
        self.file = open(output_file, "a")

    def append(self, data):
        self.file.write(json.dumps(data) + "\n")
        # print("Writer DONE")

    def truncate(self):
        with open(self.output_file, 'w'):
            pass
        
        self.file = open(self.output_file, "a")
        
    
    def close(self):
        self.file.close()
