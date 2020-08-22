import tensorflow as tf
from tensorflow import keras


class SuperviseClassModel(object):
    def __init__(self, n_output):
        self.n_output = n_output
        self.m_part1 = keras.Sequential([keras.layers.Dense(50, activation='tanh', input_shape=[100]),
                                         keras.layers.Dense(30, activation='relu'),
                                         keras.layers.Dense(50, activation='tanh'),
                                         keras.layers.Dense(100, activation='relu')])
        self.m_part2 = keras.Sequential([keras.layers.Dense(self.n_output, input_shape=[100])])
        self.model = keras.Sequential([self.m_part1, self.m_part2])

    def model_compile(self):
        self.model.compile(optimizer='rmsprop',
                           loss=keras.losses.CategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])

    def training(self, x, y):
        self.model.fit(x, y, epochs=50, batch_size=32)

    def get_embedding_vec(self, x):
        prob_m_part1 = tf.keras.Sequential([self.m_part1, tf.keras.layers.Softmax()])
        return prob_m_part1.predict(x)
