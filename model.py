import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.applications import EfficientNetB0
import numpy as np

# Define Channel Attention Layer with Residual Connection
class ChannelAttention(layers.Layer):
    def __init__(self, reduction_ratio=16, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        self.avg_pool = layers.GlobalAveragePooling2D()
        self.max_pool = layers.GlobalMaxPooling2D()
        self.mlp = tf.keras.Sequential([
            layers.Dense(input_shape[-1] // self.reduction_ratio, activation='relu'),
            layers.Dense(input_shape[-1], activation='sigmoid')
        ])

    def call(self, inputs):
        avg_pool = self.avg_pool(inputs)
        max_pool = self.max_pool(inputs)
        avg_pool = tf.expand_dims(tf.expand_dims(avg_pool, 1), 1)
        max_pool = tf.expand_dims(tf.expand_dims(max_pool, 1), 1)
        channel_attention = self.mlp(avg_pool + max_pool)
        return inputs * channel_attention + inputs  # Add residual connection

# Define Spatial Attention Layer with Residual Connection
class SpatialAttention(layers.Layer):
    def __init__(self, kernel_size=7, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)
        self.conv = layers.Conv2D(filters=1, kernel_size=kernel_size, activation='sigmoid', padding='same')

    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        concat = tf.concat([avg_pool, max_pool], axis=-1)
        attention = self.conv(concat)
        return inputs * attention + inputs  # Add residual connection

# Define Dynamic Attention Fusion Layer
class DynamicAttentionFusion(layers.Layer):
    def __init__(self, **kwargs):
        super(DynamicAttentionFusion, self).__init__(**kwargs)
        self.dense = layers.Dense(2, activation='softmax')

    def call(self, channel_features, spatial_features):
        concatenated_features = tf.concat([channel_features, spatial_features], axis=-1)
        attention_scores = self.dense(concatenated_features)
        channel_weight, spatial_weight = tf.split(attention_scores, num_or_size_splits=2, axis=-1)
        fused_features = channel_weight * channel_features + spatial_weight * spatial_features
        return fused_features

# Define Convolutional Token Embedding Layer
class ConvEmbedding(layers.Layer):
    def __init__(self, embed_dim, downsample=True, **kwargs):
        super(ConvEmbedding, self).__init__(**kwargs)
        self.conv = layers.Conv2D(embed_dim, kernel_size=3, strides=2 if downsample else 1, padding='same')
        self.norm = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.norm(x)
        return x

# Define Transformer Encoder Block with Residual Connection
class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        batch_size, height, width, channels = tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2], inputs.shape[-1]
        x = tf.reshape(inputs, (batch_size, height * width, channels))

        attn_output = self.att(x, x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return tf.reshape(out2, (batch_size, height, width, channels))


# Define the CvT Model with a single stage transformer
def CvT_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    # Single stage
    x = ConvEmbedding(embed_dim=64, downsample=True)(inputs)
    x = TransformerEncoder(embed_dim=64, num_heads=1, ff_dim=128)(x, training=True)

    # Global Average Pooling and Dense Layers
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax', dtype='float32')(x)  # Ensure output is float32

    model = models.Model(inputs, outputs)
    return model

class HAMFaceLoss(tf.keras.losses.Loss):
    def __init__(self, num_classes, s=30.0, m=0.5, t=0.3):
        super(HAMFaceLoss, self).__init__()
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.t = t
        self.W = tf.Variable(tf.random.normal([num_classes, 128]), trainable=True)  # weight matrix

    def call(self, y_true, embeddings):
        y_true = tf.cast(y_true, tf.int32)

        embeddings = tf.nn.l2_normalize(embeddings, axis=1)
        W_norm = tf.nn.l2_normalize(self.W, axis=1)

        cos_theta = tf.matmul(embeddings, W_norm, transpose_b=True)
        theta = tf.acos(tf.clip_by_value(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7))

        batch_indices = tf.range(tf.shape(y_true)[0], dtype=tf.int32)
        theta_yi = tf.gather_nd(theta, tf.stack([batch_indices, y_true], axis=1))

        one_hot = tf.one_hot(y_true, self.num_classes, dtype=tf.float32)
        masked_theta = theta + 1e6 * one_hot
        min_inter_theta = tf.reduce_min(masked_theta, axis=1)

        hardness = tf.cast(theta_yi + self.m > min_inter_theta, tf.float32)
        s_x = 1.0 - tf.cos(theta_yi)
        adaptive_margin = self.m + self.t * hardness * s_x

        theta_yi_modified = theta_yi + adaptive_margin
        cos_theta_yi = tf.cos(theta_yi_modified)

        logits = self.s * cos_theta
        indices = tf.stack([batch_indices, y_true], axis=1)
        updates = cos_theta_yi * self.s
        logits = tf.tensor_scatter_nd_update(logits, indices, updates)

        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=logits))


class L2Normalization(layers.Layer):
    def __init__(self, **kwargs):
        super(L2Normalization, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.nn.l2_normalize(inputs, axis=-1)

# Local feature extraction with channel and spatial attention
def local_feature_extraction(input_img, efficientnet):
    x = efficientnet(input_img)
    channel_attention = ChannelAttention()(x)
    spatial_attention = SpatialAttention()(x)
    return channel_attention, spatial_attention

def load_model(n_classes):

    # EfficientNet backbone
    efficientnet = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    efficientnet.trainable = False


    # Global feature extraction with CvT
    cvt_input = Input(shape=(128, 128, 3))
    cvt_model = CvT_model((128, 128, 3), num_classes=n_classes) #len(np.unique(y_train))
    cvt_features = cvt_model(cvt_input)

    # Combine local features
    input_img = Input(shape=(128, 128, 3))
    channel_attention, spatial_attention = local_feature_extraction(input_img, efficientnet)

    # Apply dynamic attention-based fusion
    fused_features = DynamicAttentionFusion()(channel_attention, spatial_attention)

    # Flatten the fused features
    fused_features = layers.Flatten()(fused_features)

    # Match dimensions using Dense layers
    local_features = layers.Dense(64, activation='relu')(fused_features)

    # Flatten cvt_features to match dimensions before fusion
    cvt_features_flat = layers.Dense(64, activation='relu')(cvt_features)

    # Apply dynamic attention-based fusion to combine local and global features
    combined_features_ = DynamicAttentionFusion()(local_features, cvt_features_flat)

    # Fully connected layers for final classification
    # x = layers.Dense(128, activation='relu')(combined_features_)
    # x = layers.Dropout(0.5)(x)
    # output = layers.Dense(len(np.unique(y_train)), activation='softmax', dtype='float32')(x)  # Ensure output is float32

    # Final embedding layer instead of classification
    x = layers.Dense(128)(combined_features_)
    embedding_output = L2Normalization()(x)  # L2-normalize as in HAMFace

    model = tf.keras.Model(inputs=[input_img, cvt_input], outputs=embedding_output)

    return model

