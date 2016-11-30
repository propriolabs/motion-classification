from __future__ import division
import sys
import json
import numpy as np
import logging
from flask import Flask, request
from elasticsearch import Elasticsearch
from keras.models import model_from_json
from utils.detect_peaks import detect_peaks
from utils.main_utils import load_classifier, check_session, add_strokes, templater, get_data, get_indices, empty_events, classify, timestamp, event_index, count_rallies, add_games, calories, rallies_to_json
from config import config

app = Flask(__name__)
es = Elasticsearch(config["es_url"] + ":" + config["es_port"])
model_name = sys.argv[-1]
logFormatStr = '[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s'
logging.basicConfig(format = logFormatStr, filename = "global.log", level=logging.ERROR)
formatter = logging.Formatter(logFormatStr, '%m-%d %H:%M:%S')
fileHandler = logging.FileHandler("summary.log")
fileHandler.setLevel(logging.INFO)
fileHandler.setFormatter(formatter)
streamHandler = logging.StreamHandler()
streamHandler.setLevel(logging.DEBUG)
streamHandler.setFormatter(formatter)
app.logger.addHandler(fileHandler)
app.logger.addHandler(streamHandler)
app.logger.info("Logging is set up.")
print("Serving ", model_name, "; Last updated ", config["last_updated"])
if model_name == 'randomforest':
    swing_classifier = load_classifier(config["rf_right"])
    swing_classifier_ll = load_classifier(config["rf_left"])
    swing_classifier_r = load_classifier(config["rf_right_moto"])
    swing_classifier_l = load_classifier(config["rf_left_moto"])
    templates = templater(config["dtw_template_right"])
    templates_left = templater(config["dtw_template_left"])
elif model_name == 'rnn':
    model_right = model_from_json(open(config["lstm_right"]).read())
    model_right.load_weights(config["lstm_right_weights"])
    model_left = model_from_json(open(config["lstm_left"]).read())
    model_left.load_weights(config["lstm_left_weights"])
elif model_name == 'cnn':
    model_right = model_from_json(open(config["cnn_right"]).read())
    model_right.load_weights(config["cnn_right_weights"])
    model_left = model_from_json(open(config["cnn_left"]).read())
    model_left.load_weights(config["cnn_left_weights"])

def classification_pipeline(session_id, user_id, time):
    data = get_data(es, config["es_index"], config["es_doc_type"], session_id, user_id, time)
    if data.empty is False:
        peaks = detect_peaks(data['a'], mph=30, mpd=17, show=False)
        if peaks.any():
            try:
                indices = get_indices(len(data['a']), peaks, buffer_size=8)
                labels_swing = ['backhand', 'forehand', 'serve']
                if model_name == 'randomforest':
                    probs = []
                    for index in indices:
                        if data.hand[0] == 'Righty':
                                probs.append(classify(templates, np.array(data[['ax', 'ay', 'az', 'g1', 'g2', 'g3', 'gx', 'gy', 'gz']]), index, swing_classifier))
                        elif data.hand[0] == 'Lefty':
                            probs.append(classify(templates_left, np.array(data[['ax', 'ay', 'az', 'g1', 'g2', 'g3', 'gx', 'gy', 'gz', 'r1', 'r2', 'r3']]), index, swing_classifier_ll))
                    predictions = map(lambda x: labels_swing[np.argmax(x)], probs)
                if model_name == 'rnn' or model_name == 'cnn':  # Slices data into chunks of 32 observations
                    if data.hand[0] == 'Righty':
                        chunks = np.array(map(lambda x: np.array(data[['ax', 'ay', 'az', 'g1', 'g2', 'g3', 'gx', 'gy', 'gz']].iloc[x[0]:x[1], :]), indices))
                        predictions = map(lambda x: labels_swing[x], model_right.predict_classes(chunks))
                    elif data.hand[0] == 'Lefty':
                        chunks = np.array(map(lambda x: np.array(data[['ax', 'ay', 'az', 'g1', 'g2', 'g3', 'gx', 'gy', 'gz']].iloc[x[0]:x[1], :]), indices))
                        predictions = map(lambda x: labels_swing[x], model_left.predict_classes(chunks))
                times = [data['watchtime'][i] for i in peaks]
                max_acceleration = data['a'].iloc[peaks]
                max_acceleration = [round(x, 1) for x in max_acceleration]
                max_velocity = data['velocity'].iloc[peaks]
                max_velocity = [round(x, 1) for x in max_velocity]
                events = zip(times, predictions, max_acceleration, max_velocity)
                counter = {"Forehands": 0,
                           "Backhands": 0,
                           "Serves": 0}
                event_counts = {'userId': user_id,
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
                                'aggregate': counter,
                                'timestamped': {
                                                'Forehands': timestamp('forehand', events),
                                                'Backhands': timestamp('backhand', events),
                                                'Serves': timestamp('serve', events)
                                                },
                                'rallies': [],
                                'max_rally': 0,
                                'calories': 0,
                                'mean_rally': 0,
                                'total_points': 0
                                }
                return json.dumps(event_counts)
            except ValueError:
                print("Value Error... Likely, one of the sensors isn't working")
        else:
            print("No peaks this time...")
            return empty_events(session_id, user_id, data)
    else:
        print("No raw data...")


@app.route('/tennis', methods=['POST'])
def tennis_analyzer():
    """
    Route on Flask Server to classify tennis strokes
    """
    if request.headers['Content-Type'] == 'application/json':
        session_id = request.json['session']
        user_id = request.json['userId']
        if session_id != '0':
            session_data = check_session(session_id, es, config["es_index"], config["es_analyzed_doc"])
            if session_data is False:
                print("Creating new session: ", session_id)
                results = classification_pipeline(session_id, user_id, 0)
                return results
            else:
                print("Using established session: ", session_id)
                time_marker = session_data["max_time"] - 10000
                results = json.loads(classification_pipeline(session_id, user_id, time_marker))
                session_data["samples"] += results["samples"]
                session_data["max_time"] = results["max_time"]

                session_data["timestamped"]["Forehands"] = add_strokes(session_data, results, "Forehands")
                session_data["timestamped"]["Backhands"] = add_strokes(session_data, results, "Backhands")
                session_data["timestamped"]["Serves"] = add_strokes(session_data, results, "Serves")

                session_data["aggregate"]["Forehands"] = len(session_data["timestamped"]["Forehands"])
                session_data["aggregate"]["Backhands"] = len(session_data["timestamped"]["Backhands"])
                session_data["aggregate"]["Serves"] = len(session_data["timestamped"]["Serves"])

                rallies = count_rallies(event_index(session_data))
                rallies["games"] = add_games(rallies)["games"]
                rallies["games_count"] = add_games(rallies)["games_count"]
                try:
                    max_rally = max(rallies.rally_count)
                    total_points = max(rallies.point_count)
                    mean_rally = round(np.mean(list(rallies.groupby(['point_count'], sort=False)['rally_count'].max())), 1)
                except ValueError:
                    max_rally = 0
                    total_points = 0
                    mean_rally = 0
                session_data["max_rally"] = max_rally
                session_data["total_points"] = total_points
                session_data["mean_rally"] = mean_rally
                session_data["calories"] = calories(rallies)
                session_data["rallies"] = rallies_to_json(rallies)
                print(session_data["userName"], session_data["aggregate"])
                return json.dumps(session_data)
        else:
            print("""Error: session_id is %s and user_id is %s \n
                     (1) Is ElasticSearch running? \n
                     (2) Is the NodeData server is running? \n
                     (3) Is the watch on?""" % (str(session_id), str(user_id)))
    else:
        print("Error: malformed 'request.headers'")


if __name__ == '__main__':

    """
    (1) Set up ElasticSearch
    (2) Load classifiers based on user's preferred model type
    (3) Serve the classifier!
    """
    logFormatStr = '[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s'
    logging.basicConfig(format = logFormatStr, filename = "global.log", level=logging.ERROR)
    formatter = logging.Formatter(logFormatStr, '%m-%d %H:%M:%S')
    fileHandler = logging.FileHandler("summary.log")
    fileHandler.setLevel(logging.INFO)
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setLevel(logging.DEBUG)
    streamHandler.setFormatter(formatter)
    app.logger.addHandler(fileHandler)
    app.logger.addHandler(streamHandler)
    app.logger.info("Logging is set up.")
    es = Elasticsearch(config["es_url"] + ":" + config["es_port"])
    try:
        model_name = sys.argv[1]
        print("Serving ", model_name, "; Last updated ", config["last_updated"])
        if model_name == 'randomforest':
            swing_classifier = load_classifier(config["rf_right"])
            swing_classifier_ll = load_classifier(config["rf_left"])
            swing_classifier_r = load_classifier(config["rf_right_moto"])
            swing_classifier_l = load_classifier(config["rf_left_moto"])
            templates = templater(config["dtw_template_right"])
            templates_left = templater(config["dtw_template_left"])
        elif model_name == 'rnn':
            model_right = model_from_json(open(config["lstm_right"]).read())
            model_right.load_weights(config["lstm_right_weights"])
            model_left = model_from_json(open(config["lstm_left"]).read())
            model_left.load_weights(config["lstm_left_weights"])
        elif model_name == 'cnn':
            model_right = model_from_json(open(config["cnn_right"]).read())
            model_right.load_weights(config["cnn_right_weights"])
            model_left = model_from_json(open(config["cnn_left"]).read())
            model_left.load_weights(config["cnn_left_weights"])
        else:
            print("ERROR: Invalid model type")
        app.run(host='0.0.0.0', port=config["analysis_port"], debug=True,threaded=True)
    except IndexError:
        print("ERROR: No model specified. Try this: \n\n python main.py randomforest \n")
