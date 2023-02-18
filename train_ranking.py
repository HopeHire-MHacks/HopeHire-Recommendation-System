import tensorflow as tf
import tensorflow_recommenders as tfrs
import numpy as np
import pandas as pd
from ast import literal_eval
from datetime import datetime
import os
import json

from models.ranking import RankingModel

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
        "bookmarked": x['bookmarked'],

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

        "patient_listing_timetable": x['patient_listing_timetable'],
    })

    # Grab unique ids for model input
    unique_patient_ids = np.unique(df['patient_id'])
    unique_listing_ids = np.unique(df['listing_id'])

    # Create train test dataset
    train_length = round(split_train_perc * len(df))

    shuffled = patients_dataset.shuffle(len(df), seed=seed, reshuffle_each_iteration=False)
    train = shuffled.take(train_length)
    test = shuffled.skip(train_length).take(len(df) - train_length)

    return unique_patient_ids, unique_listing_ids, train, test


if __name__ == '__main__':
    print("**GPU ENABLED**" if len(tf.config.list_physical_devices('GPU')) >= 1 else "**NO GPU FOUND**")

    SAVE_PATH = "./saved_models/ranking_" + datetime.now().strftime("%Y%m%d_%H%M")

    # Load data
    print("\n -- Loading data -- \n")
    unique_patient_ids, unique_listing_ids, train_data, test_data = load_data("./data/patient_listing_bookmarked.csv")

    # Preparing model
    print("\n -- Preparing model -- \n")
    ranking_model = RankingModel(unique_patient_ids, unique_listing_ids)
    ranking_model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))
    
    # Train
    cached_train = train_data.shuffle(99).batch(64).cache()
    cached_test = test_data.batch(64).cache()

    print("\n-- Starting Training -- \n")
    ranking_model.fit(cached_train, epochs=20)

    # Evaluation
    print("\n-- Starting Evaluation -- \n")
    ranking_model.evaluate(cached_test, return_dict=True)

    # Prediction
    print("\n-- Prediction -- \n")
    
    pred_input = { 
        "patient_id": np.array(["1"]), 
        "patient_age": np.array([35.9]),
        "patient_dialysis_freq": np.array([1.784]),
        "patient_dialysis_latitude": np.array([1.379]),
        "patient_dialysis_longitude": np.array([103.81]),
        "patient_skills": np.array([['public speaking'] * 78]),

        "listing_id": np.array(["2"]),
        "employer_num_employees": np.array([69.0]),
        "listing_industry_type": np.array(['Construction']),
        "listing_loc_latitude": np.array([1.348]),
        "listing_loc_longitude": np.array([103.785]),
        "listing_skills": np.array([['dancing'] * 78]),

        "patient_listing_timetable": np.array([0.5]),
    }
    score = ranking_model(pred_input)
    print(f"Rating: {score}")


    # Saving Model

    print("\n --Saving Model -- \n")
    tf.saved_model.save(ranking_model, SAVE_PATH)

    with open(os.path.join(SAVE_PATH, "example_input.json"), "w+") as f:
        json.dump({ x: pred_input[x].tolist() for x in pred_input}, f)

