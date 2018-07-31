from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


# reference can be multiple, while hypothesis is only one
def calc_bleu(refer_file, hypo_file):
    dev_ret = []
    with open(hypo_file, "r") as f:
        for i in f:
            i = i.strip()
            idxs = i.split()
            idxs = [int(idx) for idx in idxs]
            dev_ret.append(idxs)
    dev_ref = []

    with open(refer_file, "r") as f:
        for i in f:
            i = i.strip()
            idxs = i.split()
            idxs = [int(idx) for idx in idxs]
            dev_ref.append(idxs)
    bleu = 0
    smooth = SmoothingFunction()
    for ref, hyp in zip(dev_ref, dev_ret):
        ref_remove = [x for x in ref if x != 0]
        hyp_remove = [x for x in hyp if x != 0]
        s = sentence_bleu([ref_remove], hyp_remove, smoothing_function=smooth.method1)
        bleu += s
        # print(s)
    # print(i)
    # print(len(dev_ret))
    return bleu / len(dev_ret)


if __name__ == '__main__':
    dev_x = "./data/dev_idx_x.txt"
    dev_y = "./data/dev_idx_y.txt"

    test_x = "./data/test_idx_x.txt"
    test_y = "./data/test_idx_y.txt"

    dev_file = "./result/dev_ret.txt"
    test_file = "./result/test_ret.txt"

    print(calc_bleu(dev_y, dev_file))
