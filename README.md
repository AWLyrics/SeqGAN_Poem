# SeqGAN Poem
修改 SeqGAN 用于诗歌生成（去除 Oracle LSTM）

## Pipeline

- 收集唐诗预处理（preprocessing.py），唐诗[数据集](https://github.com/chinese-poetry/chinese-poetry)，选取了 5000 首五言律诗（20 词）
- Tokenize，将中文字转换为 index（train.txt），建立词表（dict.pkl），记录 vocab size
- Pretrain Generation
- Adersarial Training

## Reference

Paper: https://arxiv.org/abs/1609.05473

Original Repo: [SeqGAN](https://github.com/LantaoYu/SeqGAN)

Blog: [SeqGAN——GAN + RL + NLP](http://tobiaslee.top/2018/03/11/SeqGAN/)



