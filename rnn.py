from __future__ import absolute_import, division, print_function, unicode_literals
from sklearn.metrics import f1_score
import tensorflow as tf
import numpy as np
import json
import time

print("Number of GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

class Model(tf.keras.Model):
    def __init__(self, vocab_size):

        super(Model, self).__init__()

        self.vocab_size = vocab_size
        self.window_size = 20
        self.embedding_size = 100
        self.batch_size = 3
    
        self.E = tf.Variable(
            tf.random.truncated_normal([self.vocab_size, self.embedding_size], dtype=tf.float32, stddev=1e-1), 
            name='E')

        self.gru_layer = tf.keras.layers.GRU(128, return_sequences=False, return_state=True)
        self.dense = tf.keras.layers.Dense(2, activation='softmax')

        self.optimizer = tf.keras.optimizers.Adam(0.01)


    def call(self, inputs, initial_state=None):

        q1 = tf.nn.embedding_lookup(self.E, inputs[:,:,0])
        q2 = tf.nn.embedding_lookup(self.E, inputs[:,:,1]) # (batch, window, embedding)

        gru_out1, state1 = self.gru_layer(q1)
        gru_out2, state2 = self.gru_layer(q2) # (batch, encoding)
        h = tf.concat([gru_out1,gru_out2], axis=1) #(batch, encoding*2)
        # print(h)
        dense_out = self.dense(h)

        return dense_out

    def loss(self, logits, labels):
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=labels, y_pred=logits)
        loss = tf.reduce_mean(loss)
        return loss

def train(model, train_inputs, train_labels):
    # (N, window, 2)
    num_epochs = 10
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

        if epoch % 10 == 0:
            print("--------------------------------------------------------------")
            print('Epoch %d \t Loss: %.3f' % (epoch, train_loss / step))
            print("--------------------------------------------------------------")

def test(model, test_inputs, test_labels):
    # print(test_labels)
    logits = model.call(test_inputs)
    pred = tf.argmax(logits, axis=1)
    f1 = f1_score(pred, test_labels)
    result = tf.dtypes.cast(tf.math.equal(pred, test_labels), tf.float32)
    accuracy = tf.reduce_mean(result)
    print('Acc:', accuracy.numpy())
    print('F1 Score:', f1)
    # return accuracy


def main():
    vocab_fp = 'vocab.json'
    train_data_fp = 'train_data.npy'
    train_labels_fp = 'train_label.npy'
    test_data_fp = 'test_data.npy'
    test_labels_fp = 'test_labels.npy'
    test_data_fp = 'train_data.npy'
    test_labels_fp = 'train_label.npy'
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
    # model = Model(10)

    start = time.time()
    train(model, train_data, train_labels)
    print('Training process takes %.4f minutes' % ((time.time()-start)/60))

    accuracy = test(model, test_data, test_labels) 
    print(accuracy)
    
if __name__ == '__main__':
    main()
