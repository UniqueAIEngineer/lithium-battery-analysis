LINE = 0.986

def judge(re_first, re_now):
    ratio = re_now/re_first
    if ratio >= LINE:
        return "劣化"
    else:
        return "正常"
