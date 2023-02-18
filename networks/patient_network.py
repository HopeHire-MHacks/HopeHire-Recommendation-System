import tensorflow as tf

from constants.skills import SKILLS

class PatientNetwork(tf.keras.Model):
  
    def __init__(self, unique_patient_ids):
        super().__init__()

        self.patient_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=unique_patient_ids, mask_token=None),
            tf.keras.layers.Embedding(len(unique_patient_ids) + 1, 32), # +1 to account for the unknown token.
        ])

        self.patient_age_embedding = tf.keras.Sequential([
            tf.keras.layers.Normalization(axis=None),
            tf.keras.layers.Reshape((1, ))
        ])

        self.patient_skills_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=SKILLS, mask_token=None, output_mode="multi_hot"),
        ])

    def call(self, inputs):
        return tf.concat([
            self.patient_embedding(inputs["patient_id"]),
            self.patient_age_embedding(inputs["patient_age"]),
            tf.reshape(inputs["patient_dialysis_freq"], (-1, 1)),
            tf.reshape(inputs["patient_dialysis_latitude"], (-1, 1)),
            tf.reshape(inputs["patient_dialysis_longitude"], (-1, 1)),
            self.patient_skills_embedding(inputs["patient_skills"]),
        ], axis=-1)