import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.config.run_functions_eagerly(False)


class Model:
    def __init__(self, num_users, num_items, embedding_size):
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_size = embedding_size
        self.V = layers.Embedding(input_dim=self.num_items,
                                  output_dim=self.embedding_size,
                                  input_length=1,
                                  embeddings_initializer="he_normal",
                                  embeddings_regularizer=keras.regularizers.l2(1e-6),
                                  name="Item_Embedding")

    def Explict_MF(self, users_profiles):
        users_profiles = tf.convert_to_tensor(users_profiles)
        input = tf.keras.Input(shape=(2,), dtype=tf.int32, name='Input')
        user_id, item_id = input[:, 0], input[:, 1]
        user_embedding = layers.Embedding(input_dim=self.num_users,
                                          output_dim=self.embedding_size,
                                          input_length=1,
                                          embeddings_initializer="he_normal",
                                          embeddings_regularizer=keras.regularizers.l2(1e-6),
                                          name="User_Embedding")(user_id)

        user_vectors = layers.Flatten(name='User_Vector')(user_embedding)

        fc = layers.Dense(units=self.embedding_size, activation=tf.nn.relu, name="Profile_Weight",
                          kernel_regularizer=keras.regularizers.l2(1e-6), bias_regularizer=keras.regularizers.l2(1e-6))
        profile = tf.nn.embedding_lookup(params=users_profiles, ids=user_id)
        profile = fc(profile)
        #user_vectors = layers.Add(name="Add")([user_vectors, profile])

        item_embedding = self.V(item_id)
        item_vectors = layers.Flatten(name='Item_Vector')(item_embedding)
        output = layers.Dot(name="User_Item_Similar", axes=-1)([user_vectors, item_vectors])
        return tf.keras.Model(inputs=input, outputs=[output], name="Explict_MF")

    def Implicit_MF(self):
        input = tf.keras.Input(shape=(2,), dtype=tf.int32, name='Item')
        item_k_id, item_j_id = input[:, 0], input[:, 1]
        item_k_embedding = self.V(item_k_id)
        item_j_embedding = self.V(item_j_id)
        output = layers.Dot(name="Transition_Pattern", axes=-1)([item_k_embedding, item_j_embedding])
        output = tf.reshape(output, shape=(-1, 1))
        return tf.keras.Model(inputs=input, outputs=[output], name="Implicit_MF")
