from inference.inference import Inference
from reader.reader import Reader
from inference.pool import Pool
from prompt.prompt import Prompt
from writer.writer import Writer
from cache.cache import Cache
from cache.tiny_db import TinyDBCache
from random import randint


class Collector:
    inference: list[Inference] = list()

    pool: Pool
    prompt: list[Prompt]
    cache: Cache

    def __init__(
        self,
        inference: list[Inference],
        prompt: list[Prompt],
        pool: Pool = None,
        cache: Cache = None,
    ):
        if pool is None:
            pool = Pool()

        if inference is None:
            raise Exception("Undefined inference")

        if len(inference) == 0:
            raise Exception("inference couldn't be empty list")

        if prompt is None:
            raise Exception("Undefined prompt")

        if len(prompt) == 0:
            raise Exception("prompt couldn't be empty list")

        if cache is None:
            cache = TinyDBCache()
            print("Using {} as default cache".format(cache))

        self.cache = cache

        self.inference = inference

        self.pool = pool

        self.prompt = prompt

    def collect(self, reader: Reader, writer: Writer, slice: int = None, slice_start=0,
                random_prompt = False
                ):
        rows = reader.read()
        i = 0

        if reader.id is None:
            raise Exception("the Reader requires and id property to use the cache")

        for row in rows:
            if i < slice_start:
                i += 1
                continue

            if slice is not None and i >= slice_start + slice:
                return


            if self.cache.has(reader.id, row["id"]):
                print("SKIP [{idx}]:{id}".format(idx=i, id=row["id"]))
                i += 1
                continue

           
            print("Row[{}]".format(i))

            output = []
            selected_prompts = self.prompt
            if random_prompt is True:
                index =  randint(0, len(self.prompt) - 1)
                selected_prompts = [self.prompt[index]]
            
            for idx, prompt in enumerate(selected_prompts):
                print("\t {}".format(prompt))

                formatted_prompt = prompt.format(row)

                raw_output = self.pool.start(
                    self.inference, "completion", {"prompt": formatted_prompt}
                )

                if raw_output is None:
                    continue

                parsed_output = prompt.parse(raw_output, row)

                if parsed_output is None:
                    continue

                output.append(parsed_output)

 
            writer.append(
                {
                    # "Instruction":output[0]["instruction"],
                    # "Input": output[0]["input"] if "input" in output[0] else row["input"] ,
                    # "Response:" : output[0]["token"],
                    # "Task" : output[0]["task"],

                    "question":output[0]['question'], 
                    "context":output[0]['context'], 
                    "long_answer":output[0]['long_answer'], 
                    "final_decision":output[0]['final_decision'], 
                    "loe":output[0]['loe'],

                }
            )
            
            self.cache.put(reader.id, row["id"])

            i += 1


        writer.close()
