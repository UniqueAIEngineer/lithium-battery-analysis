from judge import judge

def test_healthy():
    assert judge(0.050, 0.049) == "正常"

def test_degraded():
    assert judge(0.050, 0.060) == "劣化"

def test_on_line():
    assert judge(1.0, 0.986) == "劣化"