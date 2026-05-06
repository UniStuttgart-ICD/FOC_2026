from voice_runtime.timing import elapsed_ms_since


def test_elapsed_ms_since_rounds_to_two_decimals() -> None:
    assert elapsed_ms_since(10.0, now=10.123456) == 123.46
