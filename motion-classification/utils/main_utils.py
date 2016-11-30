import json
import cPickle
import numpy as np
import pandas as pd
from utils.featurization import featurize


def load_classifier(filepath):

    """
    Load classifier from filepath
    """

    with open(filepath, 'rb') as fid:
        classifier = cPickle.load(fid)
    fid.close()
    return classifier


def templater(filepath):

    """
    Generate template from filepath
    """

    templates = pd.read_csv(filepath)
    return map(lambda x: np.array(x[1][['ax', 'ay', 'az']]),
               list(templates.groupby('sample')))


def get_data(es, es_index, es_doc_type, session_id, user_id, time):

    """
    Get a session's data
    """

    query = {"sort": [{
             "watchtime": {"order": "asc"}
             }],
             "query": {
                "bool": {
                    "must": [
                        {"match": {"session": session_id}},
                        {"match": {"userId": user_id}},
                        {"range": {
                            "watchtime":
                                {
                                 "gte": time
                                }
                            }
                         }
                    ]
                    }
                }
             }
    results = es.search(body=query,
                        index=es_index,
                        doc_type=es_doc_type,
                        size=100000)
    data = pd.DataFrame(map(lambda x: x['_source'], results['hits']['hits']))
    if data.empty is False:
        data['a'] = np.sqrt(data['az']**2 + data['ay']**2 + data['az']**2)
        acceleration = data['a']
        velocity = []
        for i, item in enumerate(acceleration):
            if i < 4:
                velocity.append(0)
            elif i > len(acceleration)-4:
                velocity.append(0)
            else:
                velocity.append((sum(acceleration[i-4:i+4])/8)*(0.4)*(2.23)*1.6)
        data['velocity'] = velocity
    return data


def get_indices(n_obs, peaks, buffer_size):
    indices = []
    for peak in peaks:
        # Handle edge case where peak occurs within last 17 obs
        if peak + buffer_size > n_obs or peak - buffer_size < 0:
            pass
        else:
            indices.append((peak-buffer_size, peak+buffer_size))
    return indices


def empty_events(session_id, user_id, data):

    """
    Return a JSON for when we haven't yet registered a stroke
    """

    return json.dumps({
                        'userId': user_id,
                        'session': session_id,
                        'privacy': data['privacy'].iloc[0],
                        'userName': data['userName'].iloc[0],
                        'samples': len(data['watchtime']),
                        'max_time': max(data['watchtime']),
                        'hand': data['hand'].iloc[0],
                        'bezel': data['bezel'].iloc[0],
                        'age': data['age'].iloc[0],
                        'gender': data['gender'].iloc[0],
                        'rating': data['rating'].iloc[0],
                        'heightInches': data['heightInches'].iloc[0],
                        'manufacturer': data['manufacturer'].iloc[0],
                        'model': data['model'].iloc[0],
                        'product': data['product'].iloc[0],
                        'activity': data['activity'].iloc[0],
                        'aggregate': {
                            'Backhands': 0,
                            'Forehands': 0,
                            'Serves': 0
                            },
                        'timestamped': {
                            'Forehands': [],
                            'Backhands': [],
                            'Serves':    []
                                      },
                        'rallies': [],
                        'max_rally': 0,
                        'calories': 0,
                        'mean_rally': 0,
                        'total_points': 0
                       })

def check_session(session_id, es, index, doc_type):
    query = {
           "query": {
                "match": {
                    "session": session_id}
           }
        }
    results = es.search(body=query,
                        index=index,
                        doc_type=doc_type,
                        size=1)
    if results["hits"]["total"] == 0:
        return False
    else:
        return results["hits"]["hits"][0]["_source"]


def add_strokes(session_data, results, stroke):
    try:
        times = [i["time"] for i in session_data["timestamped"][stroke]]
    except KeyError:
        times = []
    for i in results["timestamped"][stroke]:
        if i["time"] not in times:
            session_data["timestamped"][stroke].append(i)
    return session_data["timestamped"][stroke]


def classify(templates, data, index, classifier):

    """
    Featurize data and classify the features
    """

    features = featurize(data, index, templates)
    return classifier.predict_proba(features)


def count(stroke, events):

    """
    Get stroke count
    """

    return len(filter(lambda x: x[1] == stroke, events))


def timestamp(stroke, events):

    """
    Get time stamp of each stroke
    """

    stroke = map(lambda x: {"time": x[0], "max_acceleration": x[2], "max_speed": x[3]},
               filter(lambda x: x[1] == stroke, events))
    return stroke

def event_index(data):
    time = []
    stroke = []
    speed = []
    for k, v in data["timestamped"].items():
        for i in v:
            stroke.append(k)
            time.append(i["time"])
            speed.append(i["max_speed"])
    data = pd.DataFrame({"time": time, "ballspeed": speed, "stroke": stroke})
    data = data.sort(['time'], ascending=[1])
    data = data.reset_index(drop=True)
    return data

def count_rallies(data):
    serve_count = 0
    rally_count = 0
    last_time = 0
    points = []
    serves = []
    rallies =[]
    time = []
    point_count = []
    stroke = []
    counter = 0
    for row in data.iterrows():
        diff = row[1]["time"] - last_time
        last_time = row[1]["time"]
        if diff > 8000 and rally_count > 0:  # The point ends after 8 seconds of inactivity if a rally has started
            serve_count = 0
            rally_count = 0
        elif diff > 12000 and rally_count == 0:  # The point ends after 12 seconds of inactivity if a rally hasn't started
            serve_count = 0
            rally_count = 0
        if row[1]["stroke"] == "Serves":
            status = "serve"
            serve_count += 1
        if row[1]["stroke"] == "Forehands" or row[1]["stroke"] == "Backhands":
            if serve_count == 0 and rally_count == 0:
                status = "return"
            else:
                status = "rally"
            rally_count += 1
        if serve_count == 1 or rally_count == 1:
            counter += 1
        if row[1]["stroke"] == "Serves":
            stroke.append("serve")
        elif row[1]["stroke"] == "Forehands":
            stroke.append("forehand")
        elif row[1]["stroke"] == "Backhands":
            stroke.append("backhand")
        points.append(status)
        serves.append(serve_count)
        rallies.append(rally_count)
        time.append(row[1]["time"])
        point_count.append(counter)
    return pd.DataFrame({"point_count": point_count, "time": time,
                         "status": points, "serve_count": serves,
                         "rally_count": rallies, "ballspeed": data["ballspeed"], "stroke": stroke})


def rallies_to_json(rallies):
    rally_json = []
    for i in list(rallies.groupby(['point_count'])):
        rally = {}
        result = []
        data = i[1]
        for row in data.iterrows():
            rally["time"] = row[1]["time"]
            rally["ballspeed"] = row[1]["ballspeed"]
            rally["stroke"] = row[1]["stroke"]
            result.append(rally)
            rally = {}
        rally_json.append({"game_type": data["games"].iloc[0], "game_number": data["games_count"].iloc[0], "rally": result})
    return rally_json


def add_games(rallies):
    games = []
    game = 0
    games_count = []
    counter = 1
    last_status = ""
    for i in rallies.iterrows():
        if i[1]["status"] == "serve":
            game = "Service"
        if i[1]["status"] == "return":
            game = "Return"
        if game != last_status and last_status != "":
            counter += 1
        last_status = game
        games.append(game)
        games_count.append(counter)
    return {"games": games, "games_count": games_count}


def calories(rallies):
    calories = 0
    for i in rallies.iterrows():
        if i[1]["rally_count"] == 1:
            calories += 1.8 * i[1]["ballspeed"]/80
        elif i[1]["rally_count"] > 1 and i[1]["rally_count"] < 4:
            calories += 2.2 * i[1]["ballspeed"]/80
        elif i[1]["rally_count"] > 3:
            calories += 2.7 * i[1]["ballspeed"]/80
    return round(calories, 1)
