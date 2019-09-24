from ranking_layer.dssm_network import *

def train():

    train_batch = # todo

    predict = infer(train_batch)
    loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=train_batch['label'], logits=predict, pos_weight=sample_weights,
                                                                   name="sigmoid_loss"))

    auc, auc_update_op = tf.metrics.auc(labels=train_batch['label'], predictions=tf.nn.sigmoid(predict),
                                        num_thresholds=1000)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    optimizer = tf.contrib.opt.LazyAdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss, global_step=global_step)

    saver = tf.train.Saver(max_to_keep=10)

    with tf.Session() as sess:
        for i in range(TRAINING_STEP):
            tf.global_variables_initializer().run()
            _, loss, auc, _, global_step = sess.run([train_op, loss, auc, auc_update_op, global_step])
            if i % 1000 == 0:
                print('After running %s step, the loss is %g' %(global_step, loss))

if __name__ == '__main__':
    train()