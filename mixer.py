import tensorflow as tf
from tensorflow.keras import layers

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout = layers.Dropout(dropout)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        out1 = self.layernorm1(inputs + self.dropout(attn_output, training=training))
        ffn_output = self.ffn(out1)
        return self.layernorm2(out1 + self.dropout(ffn_output, training=training))


class MambaBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim):
        super(MambaBlock, self).__init__()
        self.dense1 = layers.Dense(embed_dim, activation='gelu')
        self.dense2 = layers.Dense(embed_dim)

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)


class TransformerMambaMixer(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, mode='parallel_concat'):
        super(TransformerMambaMixer, self).__init__()
        self.transformer = TransformerBlock(embed_dim, num_heads, ff_dim)
        self.mamba = MambaBlock(embed_dim)
        self.mode = mode.lower()
        if self.mode == 'parallel_concat':
            self.proj = layers.Dense(embed_dim)

    def call(self, inputs, training=False):
        if self.mode == 'sequential_mamba_first':
            x = self.mamba(inputs)
            return self.transformer(x, training=training)

        elif self.mode == 'sequential_transformer_first':
            x = self.transformer(inputs, training=training)
            return self.mamba(x)

        elif self.mode == 'parallel_sum':
            m_out = self.mamba(inputs)
            t_out = self.transformer(inputs, training=training)
            return m_out + t_out

        elif self.mode == 'parallel_concat':
            m_out = self.mamba(inputs)
            t_out = self.transformer(inputs, training=training)
            concat_out = tf.concat([m_out, t_out], axis=-1)
            return self.proj(concat_out)

        else:
            raise ValueError(f"Invalid fusion mode: {self.mode}")


class FramewiseMixer(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, mode='parallel_concat'):
        super(FramewiseMixer, self).__init__()
        self.mixer = TransformerMambaMixer(embed_dim, num_heads, ff_dim, mode)

    def call(self, inputs, training=False):
        # inputs: (batch, time, height, width, channels)
        b, t, h, w, c = tf.unstack(tf.shape(inputs))
        # Reshape to (batch * time, h * w, c)
        x = tf.reshape(inputs, (b * t, h * w, c))
        # Apply mixer
        x = self.mixer(x, training=training)
        # Reshape back to (batch, time, h, w, c)
        x = tf.reshape(x, (b, t, h, w, c))
        return x

# Example input
input_tensor = tf.random.normal((2, 40, 32, 32, 64))

# Initialize framewise mixer
framewise_mixer = FramewiseMixer(embed_dim=64, num_heads=4, ff_dim=128, mode='parallel_concat')

# Forward pass
output = framewise_mixer(input_tensor)

# Output shape: (2, 40, 32, 32, 64)
print(output.shape)
