**Proprio Motion Classification**
------------------
  Proprio Motion Classification sets up a server to classify streaming motion data into meaningful metrics. The inputs are a session id and user id. The outputs are a JSON with the associated metrics. Currently, Proprio Motion Classification is designed for transforming accelerometer and gyroscope to tennis strokes. The output will consist of information on the time and type of each stroke, rally and game information, and user and watch hardware information.

## Set up environment on Ubuntu 14.04

    ./setup.sh

## Set up and run classification server
  
    pip install -r requirements
    cd motion-classification
    cp config.py.template config.py
  
  Gunicorn

    gunicorn --bind 0.0.0.0:5000 -t 6000 -w 4 --log-level="debug" main:app randomforest

  Flask (deprecated)

    python main.py randomforest

Finally, make sure to edit the config.py for the local dev, server dev, or server production as outlined below:

### Development on Local Computer

Change the config.py file such that 

    es_url = 'localhost'
    es_index = 'proprio'
    analysis_port = 5000

### Development on Server

Change the config.py file such that 

    es_url = 'IP.Address.of.Data.Server'
    es_index = 'proprio_dev'
    analysis_port = 6000

### Development on Production

Change the config.py file such that 

    es_url = 'IP.Address.of.Data.Server'
    es_index = 'proprio'
    analysis_port = 5000

## API Details

* **URL**

  /tennnis

* **Method:**

  `POST`
  
* **Data Params**

  {'session':session_id_from_proprio_app}

* **Success Response:**

  * **Code:** 200 <br />
    **Content:** `{'rating': '4.5', 'userId': '10553...', 'session': '1463957174937', 'bezel': 'Towards Wrist', 'heightInches': '5\' 9"', 'privacy': 'No', 'samples': 321, 'total_points': 3, 'timestamped': {'Backhands': [], 'Forehands': [{'max_speed': 74.1, 'max_acceleration': 110.3, 'time': 1463957178935}, {'max_speed': 134.7, 'max_acceleration': 230.8, 'time': 1463957181836}], 'Serves': [{'max_speed': 148, 'max_acceleration': 276.6, 'time': 1463957180618}]}, 'product': 'bowfin', 'max_time': 1463957186225, 'activity': 'Tennis', 'hand': 'Lefty', 'aggregate': {'Backhands': 0, 'Forehands': 2, 'Serves': 1}, 'manufacturer': 'Motorola', 'userName': 'Superman', 'mean_rally': 1.3, 'gender': 'Male', 'age': '25', 'calories': 21.6, 'max_rally': 2, 'rallies': [{'game_number': 1, 'game_type': 'Return', 'rally': [{'ballspeed': 74.1, 'stroke': 'forehand', 'time': 1463957178935}]}, {'game_number': 2, 'game_type': 'Service', 'rally': [{'ballspeed': 148.0, 'stroke': 'serve', 'time': 1463957180618}]}, {'game_number': 2, 'game_type': 'Service', 'rally': [{'ballspeed': 134.7, 'stroke': 'forehand', 'time': 1463957181836}]}], 'model': 'Moto 360'}`
 
* **Error Response:**

  * **Code:** 404 NOT FOUND <br />
    **Content:** `{ error : "Session doesn't exist" }`

  OR

  * **Code:** 401 UNAUTHORIZED <br />
    **Content:** `{ error : "You are unauthorized to make this request." }`

* **Sample Call:**

  ```javascript
    $.ajax({
      url: "/tennis",
      dataType: "json",
      type : "POST",
      success : function(r) {
        console.log(r);
      }
    });
  ```

# Classifier

``` python classifier.py name_of_new_classifier plot ```