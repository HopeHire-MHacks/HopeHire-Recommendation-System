import tensorflow as tf
import tensorflow_recommenders as tfrs

from networks.listing_network import ListingNetwork
from networks.patient_network import PatientNetwork

class ListingPatientRetrievalModel(tfrs.models.Model):

    def __init__(self, unique_patient_ids, unique_employer_ids, patients):
        super().__init__()

        self.query_model = tf.keras.Sequential([
            ListingNetwork(unique_employer_ids),
            tf.keras.layers.Dense(32)
        ])

        self.candidate_model = tf.keras.Sequential([
            PatientNetwork(unique_patient_ids),
            tf.keras.layers.Dense(32)
        ])

        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=patients.batch(128).map(self.candidate_model),
            ),
        )

    def compute_loss(self, features, training=False):
        query_embeddings = self.query_model({ 
            "listing_id" : features["listing_id"],
            "employer_num_employees": features['employer_num_employees'],
            "listing_industry_type": features['listing_industry_type'],
            "listing_loc_latitude": features['listing_loc_latitude'],
            "listing_loc_longitude": features['listing_loc_longitude'],
            "listing_skills": features['listing_skills'],
        })

        candidate_embeddings = self.candidate_model({
            "patient_id": features["patient_id"],
            "patient_age": features["patient_age"],
            "patient_dialysis_freq": features["patient_dialysis_freq"],
            "patient_dialysis_latitude": features["patient_dialysis_latitude"],
            "patient_dialysis_longitude": features["patient_dialysis_longitude"],
            "patient_skills": features["patient_skills"],
        })

        return self.task(query_embeddings, candidate_embeddings)