from lexile_corpus_tuner.estimators.dummy_estimator import DummyLexileEstimator


def test_dummy_estimator_tracks_complexity():
    estimator = DummyLexileEstimator()
    simple = "Short words. Easy text."
    complex_text = (
        "This particularly complicated sentence contains elaborate terminology "
        "across multiple clauses."
    )

    simple_score = estimator.predict_scalar(simple)
    complex_score = estimator.predict_scalar(complex_text)

    assert simple_score > 0
    assert complex_score > simple_score
