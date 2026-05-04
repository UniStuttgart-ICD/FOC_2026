from wake.transcript_cleanup import strip_wake_phrase


def test_strips_leading_mave():
    assert strip_wake_phrase("Mave, move up a bit") == "move up a bit"


def test_strips_hey_mave():
    assert strip_wake_phrase("hey mave stop") == "stop"


def test_leaves_non_wake_text_unchanged():
    assert strip_wake_phrase("move up a bit") == "move up a bit"
