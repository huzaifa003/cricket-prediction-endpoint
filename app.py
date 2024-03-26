from flask import Flask, request, jsonify, send_from_directory
import joblib
import numpy as np

from flask import Flask, request, g, redirect, url_for, render_template, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3

import pandas as pd

import requests


app = Flask(__name__, static_url_path='', static_folder='static')

app.secret_key = 'your_secret_key'

# Load the CSV data into a pandas DataFrame
df = pd.read_csv('Batsman_Data.csv')

DATABASE = 'app.db'


# Database connection and query functions
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def create_users_table():
    conn = get_db_connection()
    conn.execute('''CREATE TABLE IF NOT EXISTS users 
                    (id INTEGER PRIMARY KEY, username TEXT UNIQUE, password TEXT)''')
    conn.commit()
    conn.close()

create_users_table()

class User(UserMixin):
    def __init__(self, id_, username):
        self.id = id_
        self.username = username

    @staticmethod
    def get(user_id):
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
        conn.close()
        if user:
            return User(id_=user['id'], username=user['username'])
        return None

@login_manager.user_loader
def load_user(user_id):
    return User.get(user_id)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        hashed_password = generate_password_hash(password)
        conn = get_db_connection()
        try:
            conn.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hashed_password))
            conn.commit()
        except sqlite3.IntegrityError:
            flash('Username already exists.')
            return redirect(url_for('signup'))
        finally:
            conn.close()
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        conn.close()
        if user and check_password_hash(user['password'], password):
            user_obj = User(id_=user['id'], username=user['username'])
            login_user(user_obj)
            return redirect(url_for('protected'))
        flash('Invalid username or password.')
        print("invalid username and password")
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))



# Load the model and model columns
rf_model = joblib.load('rf_model.pkl')
model_columns = joblib.load('model_columns.pkl')


from flask import Flask, render_template
import pandas as pd
import os

app = Flask(__name__)

# Assuming your CSV files are in the same directory as your Flask app
BOWLER_DATA_PATH = 'Bowler_data.csv'
BATSMEN_DATA_PATH = 'Batsman_Data.csv'


def clean_opposition(dataframe):
    """Remove the 'v ' prefix from the Opposition column."""
    dataframe['Opposition'] = dataframe['Opposition'].str.replace('v ', '', regex=False)
    
    return dataframe


@app.route('/prediction', methods=['GET'])
def predict_frontend():
    # Load datasets
    bowler_df = pd.read_csv(BOWLER_DATA_PATH)
    batsmen_df = pd.read_csv(BATSMEN_DATA_PATH)
    
    bowler_df = clean_opposition(bowler_df)
    batsmen_df = clean_opposition(batsmen_df)

     # Extract player names and remove duplicates by converting to a set
    bowlers = set(bowler_df['Bowler'].unique())
    batsmen = set(batsmen_df['Batsman'].unique())
    
    # Combine both sets into one sorted list
    players = sorted(bowlers.union(batsmen))
    

    # Example of extracting unique team names after cleaning, for team selection dropdown
    teams = sorted(set(bowler_df['Opposition'].unique()).union(set(batsmen_df['Opposition'].unique())))

    # Pass the combined player names to the template
    return render_template('prediction.html', players=players, teams=teams)
    



def preprocess_user_input(data):
    # One-hot encode the 'Player' and 'Opposition' fields
    encoded_data = pd.get_dummies(data, columns=['Player', 'Opposition'])
    
    # Create a DataFrame for missing columns with default value of 0
    missing_cols = {col: [0] * len(encoded_data) for col in model_columns if col not in encoded_data}
    missing_data = pd.DataFrame(missing_cols)
    
    # Concatenate the original encoded data with the missing columns DataFrame
    combined_data = pd.concat([encoded_data, missing_data], axis=1)
    
    # Ensure the order of columns matches the training data
    final_data = combined_data.reindex(columns=model_columns, fill_value=0)
    
    return final_data


@app.route('/predict', methods=['POST'])
def predict_runs():
    data = request.get_json(force=True)
    try:
        input_data = pd.DataFrame([data])
        preprocessed_input = preprocess_user_input(input_data)
        prediction = rf_model.predict(preprocessed_input)
        print(prediction[0])
        return jsonify({'predicted_runs': prediction[0], 'message': 'The Player {} is predicted to score {} runs against team {}'.format(data['Player'], prediction[0], data['Opposition'])})
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/search_batsmen', methods=['GET'])
def search_batsment():
    # Get the query parameter for search
    query = request.args.get('query', '').lower()
    
    # Check if the query string is in any column
    result_df = df.apply(lambda column: column.astype(str).str.lower().str.contains(query)).any(axis=1)
    
    # Filter the DataFrame based on the search result
    filtered_df = df[result_df]
    
    # Convert the search result to a dictionary list and return as JSON
    result = filtered_df.to_dict(orient='records')
    return jsonify(result)


@app.route('/batsmen', methods=['GET'])
def get_batsmen():
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)  # Default to 10 items per page
    
    start = (page - 1) * per_page
    end = start + per_page
    
    batsmen_data = df[start:end].to_dict(orient='records')
    return jsonify(batsmen_data)


@app.route('/batsmen_data')
def bastmen():
    return app.send_static_file('batsmen.html')



# Load the Bowler data into a pandas DataFrame
df_bowlers = pd.read_csv('Bowler_data.csv')




@app.route('/search_bowler', methods=['GET'])
def search_bowler():
    # Get the query parameter for search
    query = request.args.get('query', '').lower()
    
    # Check if the query string is in any column
    result_df = df_bowlers.apply(lambda column: column.astype(str).str.lower().str.contains(query)).any(axis=1)
    
    # Filter the DataFrame based on the search result
    filtered_df = df_bowlers[result_df]
    
    # Convert the search result to a dictionary list and return as JSON
    result = filtered_df.to_dict(orient='records')
    return jsonify(result)


@app.route('/bowler', methods=['GET'])
def get_bowler():
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)  # Default to 10 items per page
    
    start = (page - 1) * per_page
    end = start + per_page
    
    bowler_data = df_bowlers[start:end].to_dict(orient='records')
    return jsonify(bowler_data)




@app.route('/bowler_data')
def bowler_data():
    return app.send_static_file('bowler.html')


@app.route('/live_matches', methods=['GET'])
def get_live_matches():
    api_url = 'https://api.cricapi.com/v1/currentMatches?apikey=f83f0d13-0caf-4c22-b23c-1b86d52af79b&offset=0'
    response = requests.get(api_url)
    if response.status_code == 200:
        # If request was successful, return the JSON data
        return jsonify(response.json())
    else:
        # Handle errors or unsuccessful responses
        return jsonify({'error': 'Failed to fetch live matches data'}), response.status_code


@app.route("/live", methods = ['GET'])
def live():
    return app.send_static_file('live.html')


@app.route("/")
def index():
    return app.send_static_file('index.html')

if __name__ == '__main__':
    app.run(port=4000,debug=True)
