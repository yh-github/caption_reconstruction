import json
import logging
from typing import Any
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict
from evaluations.eval_vectors import VectorStats

logger = logging.getLogger(__name__)

RAW_METRIC_OBJ = dict[str, Any]

def round_metrics(metrics, ndigits=6) -> dict:
    m = {}
    for k,v in metrics.items():
        if hasattr(v, "tolist"):
            v = v.tolist()
        if isinstance(v, list) and len(v) and isinstance(v[0], float):
            m[k] = [round(x, ndigits) for x in v]
        elif isinstance(v, float):
            m[k] = round(v, ndigits)
        else:
            m[k] = v
    return m

def metrics_to_json(metrics:dict):
    try:
        return json.dumps(metrics)
    except TypeError as te:
        s = "\n".join(f"  {k}:{v.__class__.__name__}={v}" for k,v in metrics.items())
        raise Exception(f"BAD_DICT=\n{s}\n") from te


class MetricsMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)
    data_type: str
    recon_strategy: str
    video_id: str
    size: int # num_captions
    masked: list[int]


class MetricsRecord(BaseModel):
    model_config = ConfigDict(frozen=True)
    metadata: MetricsMetadata
    metrics: dict[str, VectorStats]

    def flat_metrics(self, *filter_stat:str) -> dict[str, float]:
        return {
            f"{k}_{f}":v
            for k, vs in self.metrics.items()
            for f,v in vs.model_dump().items() if not filter_stat or f in filter_stat
        }

    def to_flat_dict(self, *filter_stat:str) -> dict[str, Any]:
        d = self.metadata.model_dump()
        d.update(self.flat_metrics(*filter_stat))
        return d

class MetricsRecordRaw(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)
    metadata: MetricsMetadata
    raw_metrics: RAW_METRIC_OBJ

    def stats(self) -> MetricsRecord:
        stats_dict = {}
        for k, v in self.raw_metrics.items():
            if isinstance(v, (np.ndarray, list)):
                # Standard vector metric
                 stats_dict[k] = VectorStats.from_vector(v)
            elif isinstance(v, (float, int)):
                # Scalar metric (treat as constant stat)
                stats_dict[k] = VectorStats(mean=float(v), std=0.0, min=float(v), max=float(v))
            # Ignore strings (Base64 matrix) or other types
            
        return MetricsRecord(
            metadata=self.metadata,
            metrics=stats_dict
        )

    def stats_z_score(self, global_stats:dict[str, VectorStats]) -> MetricsRecord:
        metrics = {}
        for k, v in self.raw_metrics.items():
            if k not in global_stats:
                continue
            
            # Z-score only meaningful for vectors/scalars where we have stats
            if isinstance(v, (np.ndarray, list)):
                 val = np.array(v) if isinstance(v, list) else v
                 metrics[k] = VectorStats.from_vector((val-global_stats[k].mean)/global_stats[k].std)
            elif isinstance(v, (float, int)):
                 # Z-score of a scalar against global dist
                 z = (v - global_stats[k].mean) / global_stats[k].std if global_stats[k].std else 0.0
                 metrics[k] = VectorStats(mean=z, std=0.0, min=z, max=z)
                 
        return MetricsRecord(
            metadata=self.metadata,
            metrics=metrics
        )

