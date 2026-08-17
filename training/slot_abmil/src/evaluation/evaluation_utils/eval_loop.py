from typing import List
from src.evaluation.evaluation_utils.inference import InferenceOutput
from src.evaluation.evaluators.base_evaluator import BaseEvaluator


class EvaluationLoop:
    def __init__(self, evaluators: List[BaseEvaluator]):
        self.evaluators = evaluators

    def process_batch(self, batch_out: InferenceOutput, batch_ids: List[str]):
        for ev in self.evaluators:
            ev.process_batch(batch_out, batch_ids)

    def finalize(self):
        out = {}
        for ev in self.evaluators:
            m = ev.finalize()
            if m:
                out[ev.name] = m
        return out
