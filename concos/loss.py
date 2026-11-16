"""Contrastive Cosine Similarity (ConCoS) loss.

Matrix form (paper, Appendix A):

    L = -1 / (B(B-1)) * 1^T (Sigma ⊙ M ⊙ U U^T) 1

where U holds the L2-normalised latents, M masks the diagonal and Sigma is the
pairwise critic: +1 / -1 for same / different labels (supervised form, Eq. 2-3)
or any similarity in [-1, 1] (generalised form, Eq. 7-8).

Because only cosine similarities enter the loss, it is invariant to positive
rescaling of each latent: the radial component ||z|| is left free.
"""

import tensorflow as tf


def concos_from_critic(z: tf.Tensor, critic: tf.Tensor) -> tf.Tensor:
    """Generalised ConCoS (Eq. 7). `critic` is a (B, B) matrix with values in [-1, 1]."""
    u = tf.math.l2_normalize(z, axis=1)
    cos = tf.matmul(u, u, transpose_b=True)
    b = tf.shape(z)[0]
    mask = 1.0 - tf.eye(b, dtype=z.dtype)
    return -tf.reduce_sum(critic * cos * mask) / tf.reduce_sum(mask)


def concos_loss(z: tf.Tensor, labels: tf.Tensor) -> tf.Tensor:
    """Supervised ConCoS (Eq. 2-3): attract same-class directions, repel the rest."""
    labels = tf.reshape(labels, [-1])
    same = tf.equal(labels[:, None], labels[None, :])
    sigma = tf.where(same, 1.0, -1.0)
    return concos_from_critic(z, tf.cast(sigma, z.dtype))


def teacher_critic(features: tf.Tensor) -> tf.Tensor:
    """Pairwise cosine critic phi(x_i, x_j) from frozen teacher features (Eq. 8)."""
    f = tf.math.l2_normalize(features, axis=1)
    return tf.matmul(f, f, transpose_b=True)
