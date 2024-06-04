from inference.inference import Inference

class MockedInference(Inference):
    def completion(self, prompt: str) -> str:
        pass
