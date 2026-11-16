"""Dense encoder-decoder pairs used in the paper (Section 5.2)."""

import numpy as np
import tensorflow as tf

from .loss import concos_loss

IMAGE_SHAPE = (28, 28, 1)


def build_encoder(latent_dim: int, variational: bool) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=IMAGE_SHAPE)
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    if variational:
        z_mean = tf.keras.layers.Dense(latent_dim, name="z_mean")(x)
        z_log_var = tf.keras.layers.Dense(latent_dim, name="z_log_var")(x)
        outputs = [z_mean, z_log_var]
    else:
        outputs = tf.keras.layers.Dense(latent_dim, name="z")(x)
    return tf.keras.Model(inputs, outputs, name="encoder")


def build_decoder(latent_dim: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(latent_dim,))
    x = tf.keras.layers.Dense(256, activation="relu")(inputs)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dense(int(np.prod(IMAGE_SHAPE)), activation="sigmoid")(x)
    outputs = tf.keras.layers.Reshape(IMAGE_SHAPE)(x)
    return tf.keras.Model(inputs, outputs, name="decoder")


class ConCoSAutoencoder(tf.keras.Model):
    """AE or VAE trained with  L = recon + concos_weight * ConCoS  [+ kl_weight * KL].

    concos_weight = 0 gives the plain baselines. For the VAE, ConCoS acts on the
    posterior mean mu (deterministic), while the decoder reconstructs from a
    reparameterised sample.
    """

    def __init__(self, latent_dim=32, variational=False, concos_weight=0.5,
                 kl_weight=1e-3, **kwargs):
        super().__init__(**kwargs)
        self.variational = variational
        self.concos_weight = concos_weight
        self.kl_weight = kl_weight
        self.encoder = build_encoder(latent_dim, variational)
        self.decoder = build_decoder(latent_dim)
        self.bce = tf.keras.losses.BinaryCrossentropy()
        self.trackers = {k: tf.keras.metrics.Mean(name=k)
                         for k in ("loss", "recon", "concos", "kl")}

    @property
    def metrics(self):
        return list(self.trackers.values())

    def encode(self, x, training=False):
        """Deterministic latent: z for the AE, mu for the VAE."""
        out = self.encoder(x, training=training)
        return out[0] if self.variational else out

    def call(self, x, training=False):
        return self.decoder(self.encode(x, training), training=training)

    def _objective(self, x, y, training):
        if self.variational:
            mu, log_var = self.encoder(x, training=training)
            eps = tf.random.normal(tf.shape(mu))
            z = mu + tf.exp(0.5 * log_var) * eps
            kl = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1.0 + log_var - tf.square(mu) - tf.exp(log_var), axis=1))
        else:
            mu = z = self.encoder(x, training=training)
            kl = tf.constant(0.0)
        recon = self.bce(x, self.decoder(z, training=training))
        cc = concos_loss(mu, y)
        total = recon + self.concos_weight * cc + self.kl_weight * kl
        return {"loss": total, "recon": recon, "concos": cc, "kl": kl}

    def _update(self, losses):
        for k, v in losses.items():
            self.trackers[k].update_state(v)
        return {k: t.result() for k, t in self.trackers.items()}

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            losses = self._objective(x, y, training=True)
        grads = tape.gradient(losses["loss"], self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return self._update(losses)

    def test_step(self, data):
        x, y = data
        return self._update(self._objective(x, y, training=False))
