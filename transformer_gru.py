from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import json
import time
import tensorflow.keras.backend as K
import transformer_funcs as transformer
import os

# print("Number of GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

class Model(tf.keras.Model):
    def __init__(self, vocab_size):

        super(Model, self).__init__()

        self.vocab_size = vocab_size
        self.window_size = 150
        self.embedding_size = 100
        self.batch_size = 64
    
        self.E = tf.Variable(
            tf.random.truncated_normal([self.vocab_size, self.embedding_size], dtype=tf.float32, stddev=1e-1), 
            name='E')

        self.gru_layer = tf.keras.layers.GRU(128, return_sequences=False, return_state=True)
        # self.dense = tf.keras.layers.Dense(2, activation='softmax')
        self.dense = tf.keras.layers.Dense(2, activation=None)
        self.optimizer = tf.keras.optimizers.Adam(0.001)
        self.position = transformer.Position_Encoding_Layer(self.window_size,100)
        self.encoder = transformer.Transformer_Block(100, is_decoder=False,multi_headed=True)
        # self.dense2 = tf.keras.layers.Dense(128, activation='relu')
        
    def call(self, inputs, initial_state=None):

        q1 = tf.nn.embedding_lookup(self.E, inputs[:,:,0])
        q2 = tf.nn.embedding_lookup(self.E, inputs[:,:,1]) # (batch, window, embedding)
        # encoder_q1 = self.encoder.call(self.position.call(q1))[:,-1,:]
        # encoder_q2 = self.encoder.call(self.position.call(q2))[:,-1,:]

        encoder_q1 = self.encoder.call(self.position.call(q1))#[:,-1,:]
        encoder_q2 = self.encoder.call(self.position.call(q2))#[:,-1,:]

        #encoder_q1 = tf.reshape(encoder_q1, [-1, 128])
        #encoder_q2 = tf.reshape(encoder_q2, [-1, 128])
        # encoder_q1 = self.dense2(encoder_q1)
        # encoder_q2 = self.dense2(encoder_q2)
        gru_out1, _ = self.gru_layer(encoder_q1)
        gru_out2, _ = self.gru_layer(encoder_q2)
#        gru_out1 = tf.reduce_mean(gru_out1, axis=1)
#        gru_out2 = tf.reduce_mean(gru_out2, axis=1)
#        print(encoder_q1.shape,"@@@@@")
#        gru_out1, state1 = self.gru_layer(q1)
#        gru_out2, state2 = self.gru_layer(q2) # (batch, encoding) both
        # print(gru_out1.shape)
        h = tf.concat([gru_out1,gru_out2], axis=1) #(batch, encoding*2)
        # h = tf.concat([state1,state2], axis=1)
        # print(h)
        # h = self.dense2(h)
        logits = self.dense(h)

        # mold
#        v1 = tf.reduce_sum(tf.multiply(encoder_q1, encoder_q1), axis=1)
#        v2 = tf.reduce_sum(tf.multiply(encoder_q2, encoder_q2), axis=1)
#        r = tf.math.maximum(v1/v2, v2/v1) #(1,+)
#        logits = -tf.math.log(r-1+1e-8)
#        logits = tf.reshape(logits, [-1, 1])
#        logits = tf.concat([-logits, logits], axis=1)

        return tf.math.sigmoid(logits)

    def loss(self, logits, labels):
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=labels, y_pred=logits)
        # loss = tf.keras.losses.binary_crossentropy(y_true=labels, y_pred=logits)
        loss = tf.reduce_mean(loss)
        return loss

def train(model, train_inputs, train_labels, manager, test_inputs, test_labels):
    # (N, window, 2)
    num_epochs = 1
    train_size = train_inputs.shape[0]
    index = [i for i in range(train_size)]
    index = tf.random.shuffle(index)
    train_inputs = tf.gather(train_inputs, index)
    train_labels = tf.gather(train_labels, index)
    print('Total steps: ', int(train_size/model.batch_size))
    for epoch in range(num_epochs):
        train_loss = 0
        step = 0
        for start, end in zip(range(0, train_size - model.batch_size, model.batch_size), range(model.batch_size, train_size, model.batch_size)):
            batch_data = train_inputs[start:end]
            batch_labels = train_labels[start:end]
            with tf.GradientTape() as tape:
                logits = model.call(batch_data)
                loss = model.loss(logits, batch_labels)

            train_loss += loss
            step += 1
            if step % 50 == 0:
                print('Step %d \t Loss: %.3f' % (step, train_loss / step))
            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        if epoch % 1 == 0:
            test(model, test_inputs, test_labels)
            manager.save()
            print("--------------------------------------------------------------")
            print('Epoch %d \t Loss: %.3f' % (epoch, train_loss / step))
            print("--------------------------------------------------------------")

def test(model, test_inputs, test_labels):
    # print(test_labels)
    test_size = test_inputs.shape[0]
    print('Total steps: ', int(test_size/model.batch_size))
    count = 0
    accuracy = 0.0
    f1 = 0.0
    for start, end in zip(range(0, test_size - model.batch_size, model.batch_size), range(model.batch_size, test_size, model.batch_size)):
        cur_inputs = test_inputs[start:end]
        cur_labels = test_labels[start:end]
        logits = model.call(cur_inputs)
        pred = tf.argmax(logits, axis=1)
        f1 += f1_score(cur_labels, pred)
        result = tf.dtypes.cast(tf.math.equal(pred, cur_labels), tf.float32)
        accuracy += tf.reduce_mean(result)
        count += 1
    print('Acc:', accuracy.numpy()/count)
    print('F1 Score:', f1.numpy()/count)

def f1_score(y_true, y_pred):
    y_true = tf.dtypes.cast(K.flatten(y_true), tf.float64)
    y_pred = tf.dtypes.cast(K.flatten(y_pred), tf.float64)
    return 2*(K.sum(y_true * y_pred)+K.epsilon()) / (K.sum(y_true) + K.sum(y_pred) + K.epsilon())


def main():
    vocab_fp = 'vocab.json'
    train_data_fp = 'train_data.npy'
    train_labels_fp = 'train_label.npy'

    test_data_fp = 'test_data.npy'
    test_labels_fp = 'test_label.npy'
    with open(vocab_fp) as f:
        vocab = json.load(f)
    train_data = np.load(train_data_fp).astype(np.int32) #(N, window, 2)
    train_labels = np.load(train_labels_fp).astype(np.int32) #(N,)
    test_data = np.load(test_data_fp).astype(np.int32)
    test_labels = np.load(test_labels_fp).astype(np.int32)
    print('train data: ', train_data.shape)
    print('train labels: ', train_labels.shape)
    print('test data: ', test_data.shape)
    print('test labels: ', test_labels.shape)
    
    model = Model(len(vocab))
    # model = Model(804)

    start = time.time()
    checkpoint_dir = "./checkpoints"
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

    checkpoint = tf.train.Checkpoint(optimizer=model.optimizer, model=model)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)

    checkpoint.restore(manager.latest_checkpoint) 
    train(model, train_data, train_labels, manager, test_data, test_labels)
    print('Training process takes %.4f minutes' % ((time.time()-start)/60))

    # print(accuracy)
    
if __name__ == '__main__':
    main()
