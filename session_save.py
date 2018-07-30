import tensorflow as tf


# sess = tf.Session()
#
# saver = tf.train.Saver()
# saver.restore(sess, './seq-gan')
# inputs = [0] * 10
# print(inputs)
# # sess.run(feed_dict={G.inputs: [1, 2 ,3 ,4 ,5, 7]})

sess = tf.Session()
new_saver = tf.train.Saver()
new_saver.restore(sess, 'seq-gan.meta')
# all_vars = tf.get_collection("vars")
# for v in all_vars:
#     v_ = sess.run(v)
#     print(v_)