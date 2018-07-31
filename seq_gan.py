from mydis import Discriminator
from mygen import Generator
from dataloader import Gen_Data_loader, Dis_dataloader, Input_Data_loader
import random
import numpy as np
import tensorflow as tf
from myG_beta import G_beta
from bleu_calc import calc_bleu
import time



dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
dis_dropout_keep_prob = 0.75

EMB_DIM = 256 # embedding dimension
HIDDEN_DIM = 64  # hidden state dimension of lstm cell
SEQ_LENGTH = 20  # sequence length
START_TOKEN = 0
PRE_EPOCH_NUM = 100  # supervise (maximum likelihood estimation) epochs
PRE_DIS_NUM = 50
SEED = 88
BATCH_SIZE = 512
# vocab_size = 6915 # max idx of word token = 6914
vocab_size = 20001  # max idx of lyric token = 211 + 0(for padding)

dis_emb_size = 32

TOTAL_BATCH = 120
positive_file = './lyric.txt'
negative_file = './generator_sample.txt'
eval_file = './eval_file.txt'
generated_num = 4096
sample_time = 16  # for G_beta to get reward
num_class = 2  # 0 : fake data 1 : real data

DIS_VS_GEN_TIME = 4

x_file = "./data/train_idx_x.txt"
y_file = "./data/train_idx_y.txt"

dev_x = "./data/dev_idx_x.txt"
dev_y = "./data/dev_idx_y.txt"

test_x = "./data/test_idx_x.txt"
test_y = "./data/test_idx_y.txt"

dev_file = "./result/dev_ret.txt"
test_file = "./result/test_ret.txt"


dev_num = 1000
test_num = 1000
def main():
    # set random seed (may important to the result)
    np.random.seed(SEED)
    random.seed(SEED)

    # data loader
    # gen_data_loader = Gen_Data_loader(BATCH_SIZE)
    input_data_loader = Input_Data_loader(BATCH_SIZE)
    dis_data_loader = Dis_dataloader(BATCH_SIZE)

    D = Discriminator(SEQ_LENGTH, num_class, vocab_size, dis_emb_size, dis_filter_sizes, dis_num_filters, 0.2)
    G = Generator(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN, has_input=True)

    # avoid occupy all the memory of the GPU
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # sess
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    # change the train data to real poems  to be done
    # gen_data_loader.create_batches(positive_file)
    input_data_loader.create_batches(x_file, y_file)
    log = open('./experiment-log.txt', 'w')
    #  pre-train generator
    print('Start pre-training...')
    log.write('pre-training generator...\n')
    s = time.time()
    for epoch in range(PRE_EPOCH_NUM):
        # loss = pre_train_epoch(sess, G, gen_data_loader)
        loss = pre_train_epoch_v2(sess, G, input_data_loader)
        print("Epoch ", epoch, " loss: ", loss)
    print("pre train generator: ", time.time - s , " s")

    print("Start pre-train the discriminator")
    s = time.time()
    for _ in range(PRE_DIS_NUM):
        # generate_samples(sess, G, BATCH_SIZE, generated_num, negative_file)
        generate_samples_v2(sess, G, BATCH_SIZE, generated_num, negative_file, input_data_loader)
        # dis_data_loader.load_train_data(positive_file, negative_file)
        dis_data_loader.load_train_data(y_file, negative_file)
        for _ in range(3):
            dis_data_loader.reset_pointer()
            for it in range(dis_data_loader.num_batch):
                x_batch, y_batch = dis_data_loader.next_batch()
                feed = {
                    D.input_x: x_batch,
                    D.input_y: y_batch,
                    D.dropout_keep_prob: dis_dropout_keep_prob
                }
                _, acc = sess.run([D.train_op, D.accuracy], feed)
            # print(acc)
    print("pretrain discriminator: ", time.time - s , " s")
    g_beta = G_beta(G, update_rate=0.8)

    print('#########################################################################')
    print('Start Adversarial Training...')
    log.write('adversarial training...\n')

    for total_batch in range(TOTAL_BATCH):
        # train generator once
        s = time.time()
        for it in range(1):
            # samples = G.generate(sess)
            # print(input_data_loader.get_all().shape)
            # input_data_loader.reset_pointer()
            # samples = []
            # for i in range(input_data_loader.num_batch):
            input_x = input_data_loader.next_batch()[0]
            samples = G.generate_v2(sess, input_x)
                # print(sample)
            # print(samples)
            rewards = g_beta.get_reward(sess, samples, sample_time, D)
            feed = {G.x: samples, G.rewards: rewards, G.inputs: input_x}
            _ = sess.run(G.g_update, feed_dict=feed)
        # Test
        if total_batch % 5 == 0 or total_batch == TOTAL_BATCH - 1:
            # generate_samples(sess, G, BATCH_SIZE, generated_num, eval_file)
            # likelihood_data_loader.create_batches(eval_file)
            # test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
            avg = np.mean(np.sum(rewards, axis=1), axis=0) / SEQ_LENGTH
            buffer = 'epoch:\t' + str(total_batch) + '\treward:\t' + str(avg) + '\n'
            print('total_batch: ', total_batch, 'average reward: ', avg)
            log.write(buffer)
            print("generating dev sentences")
            dev_loader = Input_Data_loader(BATCH_SIZE)
            dev_loader.create_batches(dev_x, dev_y)
            generate_samples_v2(sess, G, BATCH_SIZE, dev_num, dev_file, dev_loader)
            bleu = calc_bleu(dev_y, dev_file)
            print("dev bleu: ", bleu)
            log.write("bleu: %.5f \n" % bleu)
        # update G_beta with weight decay
        g_beta.update_params()

        # train the discriminator
        for it in range(DIS_VS_GEN_TIME):
            # generate_samples(sess, G, BATCH_SIZE, generated_num, negative_file)
            generate_samples_v2(sess, G, BATCH_SIZE, generated_num, negative_file, input_data_loader)
            dis_data_loader.load_train_data(y_file, negative_file)

            for _ in range(3):
                dis_data_loader.reset_pointer()
                for batch in range(dis_data_loader.num_batch):
                    x_batch, y_batch = dis_data_loader.next_batch()
                    feed = {
                        D.input_x: x_batch,
                        D.input_y: y_batch,
                        D.dropout_keep_prob: dis_dropout_keep_prob
                    }
                    _ = sess.run(D.train_op, feed_dict=feed)
        print("Adversarial Epoch consumed: ", time.time() - s, " s")
    # finnal generation
    # print("Wrting final results to test_lyric file")
    # test_file = "./final2.txt"
    # generate_samples(sess, G, BATCH_SIZE, generated_num, test_file)
    print("Finished")
    log.close()
    # save model

    print("Training Finished, starting to generating test ")
    test_loader = Input_Data_loader(batch_size=BATCH_SIZE)
    test_loader.create_batches(test_x, test_y)

    generate_samples_v2(sess,G, BATCH_SIZE, test_num, test_file, test_loader)
    # saver = tf.train.Saver()
    # saver.save(sess, './seq-gan')


def generate_samples(sess, generator_model, batch_size, generated_num, output_file):
    generated_samples = []

    for i in range(generated_num // batch_size):
        one_batch = generator_model.generate(sess)
        generated_samples.extend(one_batch)
    with open(output_file, 'w') as fout:
        for poem in generated_samples:
            buffer = ' '.join([str(x) for x in poem]) + '\n'
            fout.write(buffer)


def generate_samples_v2(sess, generator_model, batch_size, generated_num, output_file, data_loader):
    generated_samples = []
    data_loader.reset_pointer()

    for i in range(generated_num // batch_size):
        target, input = data_loader.next_batch()
        one_batch = generator_model.generate_v2(sess, input)
        # print("batch:  ", one_batch)
        generated_samples.extend(one_batch)
    with open(output_file, 'w') as fout:
        for poem in generated_samples:
            buffer = ' '.join([str(x) for x in poem]) + '\n'
            fout.write(buffer)

# pre-train the Generator based on MLE method
def pre_train_epoch(sess, trainable_model, data_loader):
    # Pre-train the generator using MLE for one epoch
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in range(data_loader.num_batch):
        batch = data_loader.next_batch()
        _, g_loss = trainable_model.pretrain_step(sess, batch)
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)


def pre_train_epoch_v2(sess, trainable_model, data_loader):
    # Pre-train the generator using MLE for one epoch
    supervised_g_losses = []
    data_loader.reset_pointer()
    # print(data_loader.batch_num)
    for it in range(data_loader.num_batch):
        target, input_x = data_loader.next_batch()
        _, g_loss = trainable_model.pretrain_step_v2(sess, input_x, target)
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)


if __name__ == '__main__':
    main()
