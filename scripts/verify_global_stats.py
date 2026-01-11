
import numpy as np
from evaluations.evaluation import ReconstructionEvaluator
from evaluations.metrics import MetricsRecordRaw, MetricsMetadata

def create_mock_metric_record(score):
    # Create a minimal mock record
    meta = MetricsMetadata(
        data_type="test", 
        recon_strategy="test",
        video_id="vid_1",
        size=1,
        masked=[]
    )
    return MetricsRecordRaw(
        metadata=meta,
        raw_metrics={"my_score": score}
    )

def test_global_stats():
    # consistent scalars
    records = [
        create_mock_metric_record(0.5),
        create_mock_metric_record(0.6),
        create_mock_metric_record(0.7),
    ]
    
    print("Testing scalar metrics...")
    try:
        stats = ReconstructionEvaluator.global_stats(records)
        print("Success:", stats)
    except Exception as e:
        print("Failed:", e)

    # consistent vectors
    records_vec = [
        create_mock_metric_record(np.array([0.1, 0.2])),
        create_mock_metric_record(np.array([0.3, 0.4])),
    ]
    print("\nTesting vector metrics...")
    try:
        stats = ReconstructionEvaluator.global_stats(records_vec)
        print("Success:", stats)
    except Exception as e:
        print("Failed:", e)

    # string metrics (should be ignored or handled gracefully)
    records_str = [
        create_mock_metric_record(0.5),
        MetricsRecordRaw(
            metadata=records[0].metadata,
            raw_metrics={"my_score": 0.6, "ignored_str": "some_string"}
        )
    ]
    print("\nTesting string metrics inclusion...")
    try:
        stats = ReconstructionEvaluator.global_stats(records_str)
        print("Success:", stats)
    except Exception as e:
        print("Failed:", e)

if __name__ == "__main__":
    test_global_stats()
