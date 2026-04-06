def build_network(hidden=(64,64,64,64,64,64)):
    inp = tf.keras.Input(shape=(2,))
    x = inp
    for h in hidden:
        x = tf.keras.layers.Dense(h, activation='tanh',
                                  kernel_initializer='glorot_normal')(x)
    out = tf.keras.layers.Dense(3)(x)  # psi, p, T
    return tf.keras.Model(inp, out)