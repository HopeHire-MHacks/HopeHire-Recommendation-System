import tensorflow as tf

from constants.industry_types import INDUSTRY_TYPES
from constants.skills import SKILLS

class ListingNetwork(tf.keras.Model):
  
    def __init__(self, unique_listing_ids):
        super().__init__()

        self.listing_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=unique_listing_ids, mask_token=None),
            tf.keras.layers.Embedding(len(unique_listing_ids) + 1, 32) # +1 to account for the unknown token.
        ])

        self.industry_type_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=INDUSTRY_TYPES, mask_token=None),
            tf.keras.layers.Embedding(len(INDUSTRY_TYPES) + 1, 32) # +1 to account for the unknown token.
        ])

        self.listing_skills_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=SKILLS, mask_token=None, output_mode="multi_hot"),
        ])

    def call(self, inputs):
        return tf.concat([
            self.listing_embedding(inputs['listing_id']),
            self.industry_type_embedding(inputs['listing_industry_type']),
            tf.reshape(inputs['employer_num_employees'], (-1, 1)),
            tf.reshape(inputs["listing_loc_latitude"], (-1, 1)),
            tf.reshape(inputs["listing_loc_longitude"], (-1, 1)),
            self.listing_skills_embedding(inputs['listing_skills']),
        ], axis=-1)