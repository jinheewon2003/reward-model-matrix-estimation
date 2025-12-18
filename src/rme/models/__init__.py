from rme.models.bradley_terry import BradleyTerryModel
from rme.models.diff_pairwise_baseline import DiffPairwiseBaselineModel
from rme.models.pairwise_to_response import PairwiseToResponseModel
from rme.models.score_diff_mf import ScoreDiffMFModel
from rme.models.simple_me import SimpleMEModel

MODEL_REGISTRY = {
    "bradley_terry": BradleyTerryModel,
    "simple_me": SimpleMEModel,
    "me_pairwise_to_response": PairwiseToResponseModel,
    "me_score_diff_mf": ScoreDiffMFModel,
    "me_diff_pairwise_baseline": DiffPairwiseBaselineModel,
}

__all__ = [
    "BradleyTerryModel",
    "SimpleMEModel",
    "PairwiseToResponseModel",
    "ScoreDiffMFModel",
    "DiffPairwiseBaselineModel",
    "MODEL_REGISTRY",
]