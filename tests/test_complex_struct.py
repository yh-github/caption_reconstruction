from data_models.complex_struct import VideoAnalysis

# Example usage and testing
def test_video_analysis():
    # Example valid data
    sample_data = {
        "segments": [
            {
                "start": "00:15:58.000",
                "end": "00:16:05.000",
                "entities": ["person", "dog"],
                "objects": ["ball", "fence", "trees"],
                "stuff": ["grass", "dirt"],
                "primary_activity": "person playing fetch with dog",
                "segment_summary": "person and dog playing fetch in a park setting",
                "key_moments": [
                    {
                        "start": "00:15:58.000",
                        "end": "00:15:59.500",
                        "caption": "person winds up and throws red ball across field"
                    },
                    {
                        "start": "00:16:02.000",
                        "end": "00:16:03.500",
                        "caption": "dog leaps and catches ball mid-air"
                    }
                ]
            }
        ]
    }

    analysis = VideoAnalysis(**sample_data)
    assert analysis is not None
    assert len(analysis.segments) == 1

    segment = analysis.segments[0]
    assert segment.start == "00:15:58.000"
    assert segment.end == "00:16:05.000"
    assert segment.primary_activity == "person playing fetch with dog"
    assert len(segment.key_moments) == 2
