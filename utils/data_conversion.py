from datetime import datetime
import dateutil.parser
import numpy as np

from constants.skills import SKILLS
from constants.industry_types import INDUSTRY_TYPES

def get_age_from(dob):
    dob_dt = dateutil.parser.isoparse(dob)
    return datetime.now(dob_dt.tzinfo).year - dob_dt.year

def get_skills(skills, max_size=78):
    return list(map(lambda x: SKILLS[x], skills)) + [''] * (max_size - len(skills))

def get_industry_type(industry_type):
    return INDUSTRY_TYPES[industry_type]