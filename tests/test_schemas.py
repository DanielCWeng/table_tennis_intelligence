from ttintel.schemas import Estimate, InferenceType, Point2D, Visibility, to_dict


def test_estimate_preserves_provenance_and_serialises() -> None:
    value = Estimate(
        value=Point2D(12.0, 14.0),
        confidence=0.7,
        source="fixture",
        visibility=Visibility.PARTIAL,
        inference_type=InferenceType.MODEL_INFERRED,
        quality_flags=["occluded"],
    )
    payload = to_dict(value)
    assert payload["value"] == {"x": 12.0, "y": 14.0}
    assert payload["confidence"] == 0.7
    assert payload["source"] == "fixture"
    assert payload["inference_type"] == "model_inferred"
    assert payload["quality_flags"] == ["occluded"]
