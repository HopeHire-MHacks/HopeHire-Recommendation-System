import tensorflow as tf

from networks.listing_network import ListingNetwork
from networks.patient_network import PatientNetwork

class RankingNetwork(tf.keras.Model):

    def __init__(self, unique_patient_ids, unique_listing_ids):
        super().__init__()

        self.patient_model = PatientNetwork(unique_patient_ids)
        self.employer_model = ListingNetwork(unique_listing_ids)

        # Compute predictions.
        self.ratings = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1)
        ])

    def call(self, inputs):
        patient_embedding = self.patient_model({
            "patient_id": inputs["patient_id"],
            "patient_age": inputs['patient_age'],
            "patient_dialysis_freq": inputs['patient_dialysis_freq'],
            "patient_dialysis_latitude": inputs['patient_dialysis_latitude'],
            "patient_dialysis_longitude": inputs['patient_dialysis_longitude'],
            "patient_skills": inputs["patient_skills"],
        })

        employer_embedding = self.employer_model({
            "listing_id": inputs["listing_id"],
            "employer_num_employees": inputs['employer_num_employees'],
            "listing_industry_type": inputs['listing_industry_type'],
            "listing_loc_latitude": inputs['listing_loc_latitude'],
            "listing_loc_longitude": inputs['listing_loc_longitude'],
            "listing_skills": inputs['listing_skills'],
        })

        return self.ratings(tf.concat([
            patient_embedding, 
            employer_embedding, 
            tf.reshape(inputs['patient_listing_timetable'], (-1, 1)),
        ], axis=1))