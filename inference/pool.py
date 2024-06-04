from typing import Any
from time import time, sleep


class Pool:
    ## seconds * minutes
    ## rate limiting
    rate_limit_time = 60 * 1
    max_requests = 50

    executors: dict[Any, Any] = dict()

    def __init__(self) -> None:
        pass

    def pick(self, executors: list[Any]):
        candidate = None
        requests_count = 0

        for executer in executors:

            if executer not in self.executors:
                self.executors[executer] = {"count": 0, "first_request_time": time()}
                candidate = executer
                break
            else:
                if (
                    candidate is None
                    or self.executors[executer]["count"] < requests_count
                ):
                    candidate = executer
                    requests_count = self.executors[executer]["count"]

        if candidate is None:
            return None

        if requests_count >= self.max_requests:

            if time() - self.executors[executer]["first_request_time"] > self.rate_limit_time:
                self.executors[executer]["count"] = 0
                self.executors[executer]["first_request_time"] = time()
            else:
                return

        self.executors[executer]["count"] += 1

        return candidate

    def start(self, executors: list[Any], fn: str, args: Any):
        while True:
            executer = self.pick(executors)
            if executer is None:
                print("\twaiting for rate limiter to cooldown")
                sleep(1)
                continue

            print("\tExecuting using {}".format(executer))
            method = getattr(executer, fn)

            return method(**args)
