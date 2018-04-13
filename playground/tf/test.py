import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data


class Data:
    def __init__(self):
        self.mnist = self.get_data()

    def get_data(self):
        mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
        return mnist


class DNN:

    def __init__(self):
        self.n_nodes_hl1 = 500
        self.n_nodes_hl2 = 500
        self.n_nodes_hl3 = 500
        self.n_classes = 10
        self.batch_size = 100

        self.x = tf.placeholder('float', [None, 784])
        self.y = tf.placeholder('float')

        hidden_1_layer = {'weights': tf.Variable(tf.random_normal([784, self.n_nodes_hl1])),
                          'biases': tf.Variable(tf.random_normal([self.n_nodes_hl1]))}

        hidden_2_layer = {'weights': tf.Variable(tf.random_normal([self.n_nodes_hl1, self.n_nodes_hl2])),
                          'biases': tf.Variable(tf.random_normal([self.n_nodes_hl2]))}

        hidden_3_layer = {'weights': tf.Variable(tf.random_normal([self.n_nodes_hl2, self.n_nodes_hl3])),
                          'biases': tf.Variable(tf.random_normal([self.n_nodes_hl3]))}

        output_layer = {'weights': tf.Variable(tf.random_normal([self.n_nodes_hl3, self.n_classes])),
                        'biases': tf.Variable(tf.random_normal([self.n_classes]))}

        l1 = tf.add(tf.matmul(self.x, hidden_1_layer['weights']), hidden_1_layer['biases'])
        l1 = tf.nn.relu(l1)

        l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
        l2 = tf.nn.relu(l2)

        l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
        l3 = tf.nn.relu(l3)

        self.output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

    def train_predict(self, input_data):
        output = self.output
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=self.y))
        optimizer = tf.train.AdamOptimizer().minimize(loss)
        hm_epochs = 1
        with tf.Session() as sess:
            # sess.run(tf.global_variables_initializer())
            for epoch in range(hm_epochs):
                epoch_loss = 0
                for _ in range(int(input_data.train.num_examples / self.batch_size)):
                    epoch_x, epoch_y = input_data.train.next_batch(self.batch_size)
                    _, c = sess.run([optimizer, loss], feed_dict={self.x: epoch_x, self.y: epoch_y})
                    epoch_loss += c
                print('Epoch', epoch, 'loss', epoch_loss)

            correct = tf.equal(tf.arg_max(output, 1), tf.arg_max(self.y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct),'float')
            print('Accuracy', accuracy.eval({self.x: input_data.test.images, self.y: input_data.test.labels}))


    def predict(self):
        return self.output


def main():
    data = Data().mnist
    m = DNN()
    m.train_predict(data)
    print(data)


if __name__ == '__main__':
    main()
