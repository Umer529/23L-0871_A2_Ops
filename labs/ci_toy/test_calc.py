from calc import add, is_ai


def test_add():
    assert add(2, 3) == 5   # RED run: change 5 to 6, push, watch it fail; change back for GREEN


def test_threshold():
    assert is_ai(0.8) and not is_ai(0.2)
