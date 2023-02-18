import numpy as np
import tensorflow as tf
import requests
import json
import pandas as pd
import os

from utils.data_conversion import get_skills, get_age_from, get_industry_type

import flask
app = flask.Flask(__name__)

BACKEND_URL = "http://localhost:8000"

LISTING_PATIENT_RETRIEVAL_MODEL = "saved_models/listing_patient_retrieval_20230218_1316"
PATIENT_LISTING_RETRIEVAL_MODEL = "saved_models/patient_listing_retrieval_20230218_1036"
RANKING_MODEL = "saved_models/ranking_20230218_1055"

def send_get_request(url, *argv):
    return send_get_request_with_form(url, None, *argv)

def send_get_request_with_form(url, data, *argv):
    return requests.get(url, headers={'x-auth-token': os.getenv("TOKEN")}, data=data, *argv)

@app.route("/patients/<int:patient_id>", methods=["GET"])
def predict_patient(patient_id):
    data = { 
        "success": True, 
        "patient_id": patient_id,
        "listing_ids": [],
    }

    req_data = send_get_request(f"{BACKEND_URL}/employees/{patient_id}").content
    patient = json.loads(req_data)['data']
    
    model = tf.saved_model.load(PATIENT_LISTING_RETRIEVAL_MODEL)
    
    _, listing_ids = model({
        "patient_id": np.array([str(patient_id)]), 
        "patient_age": np.array([get_age_from(patient['dateOfBirth'])], dtype=np.int64),
        "patient_dialysis_freq": np.array([patient['dialysisFrequency']], dtype=np.float32),
        "patient_dialysis_latitude": np.array([patient['preferredLocation'][0]], dtype=np.float32),
        "patient_dialysis_longitude": np.array([patient['preferredLocation'][1]], dtype=np.float32),
        "patient_skills": np.array([get_skills(patient['skills'])]),
    }) 
    listing_ids = listing_ids.numpy().flatten()
    tf.keras.backend.clear_session()    

    req_ids = [int(float(x.decode('utf-8'))) for x in listing_ids.tolist()]

    req_data = send_get_request_with_form(f"{BACKEND_URL}/jobs/mass/ids", data={ "ids": req_ids }).content
    listings = json.loads(req_data)['data']  
    listings_df = pd.DataFrame(listings)

    model = tf.saved_model.load(RANKING_MODEL)
    
    scores = model({
        "patient_id": np.array([str(patient_id)] * len(listings)), 
        "patient_age": np.array([get_age_from(patient['dateOfBirth'])] * len(listings), dtype=np.float32),
        "patient_dialysis_freq": np.array([patient['dialysisFrequency']] * len(listings), dtype=np.float32),
        "patient_dialysis_latitude": np.array([patient['preferredLocation'][0]] * len(listings), dtype=np.float32),
        "patient_dialysis_longitude": np.array([patient['preferredLocation'][1]] * len(listings), dtype=np.float32),
        "patient_skills": np.array([get_skills(patient['skills'])[:2]] * len(listings)),

        "listing_id": listings_df['id'].astype(str),
        "employer_num_employees": listings_df['employer'].apply(lambda x: x['numberOfEmployees']).astype(np.float32),
        "listing_industry_type": listings_df['industryType'].apply(lambda x: get_industry_type(x)).astype(str),
        "listing_loc_latitude": listings_df['address'].apply(lambda x: x[0]).astype(np.float32),
        "listing_loc_longitude": listings_df['address'].apply(lambda x: x[1]).astype(np.float32),
        "listing_skills": np.array(listings_df['skills'].apply(lambda x: get_skills(x)[:2]).tolist()),

        "patient_listing_timetable": np.array([0.5] * len(listings), dtype=np.float32),
    }) 
    scores = scores.numpy().flatten()
    tf.keras.backend.clear_session()

    req_ids = np.array(req_ids)[np.argsort(scores)]
    data['listing_ids'] = req_ids.tolist()
    
    return flask.jsonify(data)

@app.route("/employers/<int:employer_id>", methods=["GET"])
def predict_employer(employer_id):
    data = { 
        "success": True, 
        "employer_id": employer_id,
        "patient_ids": [],
    }

    req_data = send_get_request(f"{BACKEND_URL}/employers/{employer_id}").content
    employer = json.loads(req_data)['data']

    req_data = send_get_request(f"{BACKEND_URL}/employers/{employer_id}/jobs").content
    listings = json.loads(req_data)['data']
    listings_df = pd.DataFrame(listings)

    model = tf.saved_model.load(LISTING_PATIENT_RETRIEVAL_MODEL)
    _, patient_ids = model({
        "listing_id": listings_df['id'].apply(lambda x: str(x)).astype(str), 
        "employer_num_employees": np.array([employer['numberOfEmployees']] * len(listings), dtype=np.float32),
        "listing_industry_type": listings_df['industryType'].apply(lambda x: get_industry_type(x)).astype(str),
        "listing_loc_latitude": listings_df['address'].apply(lambda x: x[0]).astype(np.float32),
        "listing_loc_longitude": listings_df['address'].apply(lambda x: x[1]).astype(np.float32),
        "listing_skills": np.array(listings_df['skills'].apply(lambda x: get_skills(x)).tolist()),
    }) 
    patient_ids = np.unique(patient_ids.numpy().flatten())
    tf.keras.backend.clear_session()

    req_ids = [int(float(x.decode('utf-8'))) for x in patient_ids.tolist()]
    
    req_data = send_get_request_with_form(f"{BACKEND_URL}/employee/mass/ids", data={ "ids": patient_ids.tolist() }).content
    print(req_data)
    patients = json.loads(req_data)['data']  
    patients_df = pd.DataFrame(patients)

    model = tf.saved_model.load(RANKING_MODEL)
    scores = model({
        "patient_id": np.array(patient_ids), 
        "patient_age": patients_df['dateOfBirth'].apply(get_age_from).astype(np.float32),
        "patient_dialysis_freq": patients_df['dateOfBirth'].astype(np.float32),
        "patient_dialysis_latitude": patients_df['preferredLocation'].apply(lambda x: x[0]).astype(np.float32),
        "patient_dialysis_longitude": patients_df['preferredLocation'].apply(lambda x: x[1]).astype(np.float32),
        "patient_skills": patients_df['skills'].apply(get_skills),

        "listing_id": listings_df,
        "employer_num_employees": np.array([employer['numberOfEmployees']] * len(listings)).astype(np.float32),
        "listing_industry_type": listings_df['industryType'].apply(lambda x: get_industry_type(x)).astype(str),
        "listing_loc_latitude": listings_df['address'].apply(lambda x: x[0]).astype(np.float32),
        "listing_loc_longitude": listings_df['address'].apply(lambda x: x[1]).astype(np.float32),
        "listing_skills": np.array(listings_df['skills'].apply(lambda x: get_skills(x)).tolist()),

        "patient_listing_timetable": np.array(patients_df['timetable_overlap'], dtype=np.float32),
    }) 
    scores = scores.numpy().flatten()
    tf.keras.backend.clear_session()

    patient_ids = patient_ids[np.argsort(scores)]
    data['patient_ids'] = list(map(lambda x: int(x.decode('utf-8')), patient_ids.tolist()))

    
    return flask.jsonify(data)


if __name__ == "__main__":
    app.run(host='0.0.0.0')