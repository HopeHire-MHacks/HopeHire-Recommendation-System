import tensorflow as tf
import tensorflow_recommenders as tfrs

from typing import Dict, Text

from networks.ranking_network import RankingNetwork

class RankingModel(tfrs.models.Model):

    def __init__(self, unique_patient_ids, unique_listing_ids):
        super().__init__()
        self.ranking_model: tf.keras.Model = RankingNetwork(unique_patient_ids, unique_listing_ids)
        self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
        loss = tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.RootMeanSquaredError()]
        )

    def call(self, features: Dict[str, tf.Tensor]) -> tf.Tensor:
        return self.ranking_model(features)

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        labels = features.pop("bookmarked")

        rating_predictions = self(features)

        # The task computes the loss and the metrics.
        return self.task(labels=labels, predictions=rating_predictions)