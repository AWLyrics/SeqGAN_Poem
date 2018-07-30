from nltk.translate.bleu_score import  sentence_bleu, corpus_bleu

# reference can be multiple, while hypothesis is only one
def calc_bleu(refer_file, hypo_file):
    dev_ret = []
    with open(hypo_file, "r") as f:
        for i in f:
            i = i.strip()
            idxs = i.split()
            idxs = [ int(idx) for idx in idxs]
            dev_ret.append(idxs)
    dev_ref = []

    with open(refer_file, "r") as f:
        for i in f:
            i = i.strip()
            idxs = i.split()
            idxs = [int(idx) for idx in idxs]
            dev_ref.append(idxs)
    bleu = 0
    for ref, hyp in zip(dev_ref, dev_ret):
        bleu += sentence_bleu([ref], hyp)
    return bleu / len(dev_ret)
