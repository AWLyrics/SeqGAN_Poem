from pypinyin import pinyin, FINALS, FINALS_TONE


def rhyme1(a, b):
    # 韵母相同，声调可以不同,押输出True,不押输出False
    return pinyin(a, style=FINALS) == pinyin(b, style=FINALS)


def rhyme2(a, b):
    # 韵母相同，声调也要相同,押输出True,不押输出False
    return pinyin(a, style=FINALS_TONE) == pinyin(b, style=FINALS_TONE)


if __name__ == '__main__':
    print(rhyme1('你好', '一奥'))
    print(rhyme2('你好', '一奥'))

