import tensorflow as tf
import tensorflow_recommenders as tfrs
import numpy as np
import pandas as pd
from ast import literal_eval
from datetime import datetime
import os
import json

from models.patient_listing_retrieval import PatientListingRetrievalModel

def load_data(csv_path, split_train_perc=0.8, seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    df = pd.read_csv(csv_path)
    df['patient_id'] = df['patient_id'].astype(str)
    df['listing_id'] = df['listing_id'].astype(str)

    df['employer_num_employees'] = df['employer_num_employees'].astype(np.float64)

    df['patient_skills'] = df['patient_skills'].apply(literal_eval)
    df['listing_skills'] = df['listing_skills'].apply(literal_eval)

    dataset = tf.data.Dataset.from_tensor_slices(df.to_dict(orient="list"))
    patients_dataset = dataset.map(lambda x: {
        "patient_id": x['patient_id'],
        "listing_id": x['listing_id'],
        "patient_age": x['patient_age'],
        "patient_dialysis_freq": x['patient_dialysis_freq'],
        "patient_dialysis_latitude": x['patient_dialysis_latitude'],
        "patient_dialysis_longitude": x['patient_dialysis_longitude'],
        "patient_skills": x["patient_skills"],
        "employer_num_employees": x['employer_num_employees'],
        "listing_industry_type": x['listing_industry_type'],
        "listing_loc_latitude": x['listing_loc_latitude'],
        "listing_loc_longitude": x['listing_loc_longitude'],
        "listing_skills": x['listing_skills'],
    })

    # Grab unique ids for model input
    unique_patient_ids = np.unique(df['patient_id'])

    # Create train test dataset
    train_length = round(split_train_perc * len(df))

    shuffled = patients_dataset.shuffle(len(df), seed=seed, reshuffle_each_iteration=False)
    train = shuffled.take(train_length)
    test = shuffled.skip(train_length).take(len(df) - train_length)

    return unique_patient_ids, train, test

def load_listing_data(csv_path):
    df = pd.read_csv(csv_path)

    df['listing_id'] = df['listing_id'].astype(str)
    df['employer_num_employees'] = df['employer_num_employees'].astype(np.float64)

    df['listing_skills'] = df['listing_skills'].apply(literal_eval)

    unique_listing_ids = np.unique(df['listing_id'])

    dataset = tf.data.Dataset.from_tensor_slices(df.to_dict(orient="list"))
    listings_dataset = dataset.map(lambda x: {
        "listing_id": x['listing_id'],
        "listing_skills": x['listing_skills'],
        "employer_num_employees": x['employer_num_employees'],
        "listing_industry_type": x['listing_industry_type'],
        "listing_loc_latitude": x['listing_loc_latitude'],
        "listing_loc_longitude": x['listing_loc_longitude'],
    })

    return unique_listing_ids, listings_dataset

def load_patient_data(csv_path):
    df = pd.read_csv(csv_path)

    df['patient_id'] = df['patient_id'].astype(str)
    df['patient_skills'] = df['patient_skills'].apply(literal_eval)

    unique_patient_ids = np.unique(df['patient_id'])

    dataset = tf.data.Dataset.from_tensor_slices(df.to_dict(orient="list"))
    patients_dataset = dataset.map(lambda x: {
        "patient_id": x['patient_id'],
        "patient_age": x['patient_age'],
        "patient_dialysis_freq": x['patient_dialysis_freq'],
        "patient_dialysis_latitude": x['patient_dialysis_latitude'],
        "patient_dialysis_longitude": x['patient_dialysis_longitude'],
        "patient_skills": x["patient_skills"],
    })

    return unique_patient_ids, patients_dataset

if __name__ == '__main__':
    print("**GPU ENABLED**" if len(tf.config.list_physical_devices('GPU')) >= 1 else "**NO GPU FOUND**")

    SAVE_PATH = "./saved_models/patient_listing_retrieval_" + datetime.now().strftime("%Y%m%d_%H%M")

    # Load data
    print("\n -- Loading data -- \n")
    unique_patient_ids, train_data, test_data = load_data("./data/patient_to_listings_full.csv")
    unique_listing_ids, listings_dataset = load_listing_data("./data/patient_to_listings_left.csv")

    # Preparing model
    print("\n -- Preparing model -- \n")
    retrieval_model = PatientListingRetrievalModel(unique_patient_ids, unique_listing_ids, listings_dataset)
    retrieval_model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))
    
    # Train
    cached_train = train_data.shuffle(99).batch(64).cache()
    cached_test = test_data.batch(64).cache()

    print("\n-- Starting Training -- \n")
    retrieval_model.fit(cached_train, epochs=3)

    # Evaluation
    print("\n-- Starting Evaluation -- \n")
    retrieval_model.evaluate(cached_test, return_dict=True)

    # Prediction
    print("\n-- Prediction -- \n")

    # Create a model that takes in raw query features, and
    index = tfrs.layers.factorized_top_k.BruteForce(retrieval_model.query_model)
    # recommends movies out of the entire dataset.
    index.index_from_dataset(
        tf.data.Dataset.zip((listings_dataset.map(lambda x: x["listing_id"]).batch(100), listings_dataset.batch(100).map(retrieval_model.candidate_model)))
    )

    # Get recommendations.
    pred_input = { 
        "patient_id": np.array(["1"]), 
        "patient_age": np.array([55]),
        "patient_dialysis_freq": np.array([3.0]),
        "patient_dialysis_latitude": np.array([1.405]),
        "patient_dialysis_longitude": np.array([103.901]),
        "patient_skills": np.array([["dancing"] * 78]),
    }
    _, listing_ids = index(pred_input)
    print(f"Recommendations for user 1: {listing_ids[0, :10]}")

    # Saving Model

    print("\n --Saving Model -- \n")
    tf.saved_model.save(index, SAVE_PATH)

    with open(os.path.join(SAVE_PATH, "example_input.json"), "w+") as f:
        json.dump({ x: pred_input[x].tolist() for x in pred_input}, f)
