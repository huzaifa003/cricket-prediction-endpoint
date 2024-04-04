import joblib

from flask import Flask, session, redirect, url_for, request, flash, render_template,jsonify, send_from_directory
from functools import wraps

from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3


import requests

import pandas as pd

from flask_cors import CORS

app = Flask(__name__, static_url_path='', static_folder='static')

# After app is created
CORS(app)
DATABASE = 'app.db'
app.secret_key = 'your_secret_key'

# Load the CSV data into a pandas DataFrame
df = pd.read_csv('Batsman_Data.csv')

def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def create_users_table():
    conn = get_db_connection()
    conn.execute('''CREATE TABLE IF NOT EXISTS users 
                    (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE, password TEXT)''')
    conn.commit()
    conn.close()

create_users_table()

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        print("hello")
        if 'username' not in session:
            flash('You need to be logged in to view this page.')
            print("Username not in session")
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@login_required
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']  # In a real application, you should hash the password
        
        try:
            conn = get_db_connection()
            conn.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, password))
            conn.commit()
        except sqlite3.IntegrityError:
            flash('Username already exists. Try a different one.')
            return redirect(url_for('signup'))
        finally:
            conn.close()
        
        flash('Signup successful. Please login.')
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
        
        if user is None:
            flash('Username not found.')
        elif not user['password'] == password:
            flash('Password is incorrect.')
        else:
            session['username'] = user['username']
            print("Setting username in session")
            session['username'] = user['username']
            print("Username set in session:", session['username'])

            return redirect(url_for('index'))
    return render_template('login.html')







@login_required
@app.route('/logout')
def logout():
    print(session.pop('username', None))
    return redirect(url_for('index'))






# Load the model and model columns
rf_model = joblib.load('rf_model.pkl')
model_columns = joblib.load('model_columns.pkl')



# Assuming your CSV files are in the same directory as your Flask app
BOWLER_DATA_PATH = 'Bowler_data.csv'
BATSMEN_DATA_PATH = 'Batsman_Data.csv'


def clean_opposition(dataframe):
    """Remove the 'v ' prefix from the Opposition column."""
    dataframe['Opposition'] = dataframe['Opposition'].str.replace('v ', '', regex=False)
    
    return dataframe

@login_required
@app.route('/prediction', methods=['GET'])
def predict_frontend():
    if 'username' not in session:
        flash('You need to be logged in to view this page.')
        return redirect(url_for('login'))
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
    if 'username' not in session:
        flash('You need to be logged in to view this page.')
        return jsonify({"message": "User Not logged in"})
    
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
    if 'username' not in session:
        flash('You need to be logged in to view this page.')
        return redirect(url_for('login'))
    # Get the query parameter for search
    query = request.args.get('query', '').lower()
    
    # Check if the query string is in any column
    result_df = df.apply(lambda column: column.astype(str).str.lower().str.contains(query)).any(axis=1)
    
    # Filter the DataFrame based on the search result
    filtered_df = df[result_df]
    
    # Convert the search result to a dictionary list and return as JSON
    result = filtered_df.to_dict(orient='records')
    return jsonify(result)


@login_required
@app.route('/batsmen', methods=['GET'])
def get_batsmen():
    if 'username' not in session:
        flash('You need to be logged in to view this page.')
        return redirect(url_for('login'))
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)  # Default to 10 items per page
    
    start = (page - 1) * per_page
    end = start + per_page
    
    batsmen_data = df[start:end].to_dict(orient='records')
    return jsonify(batsmen_data)


@app.route('/batsmen_data')
def bastmen():
    if 'username' not in session:
        flash('You need to be logged in to view this page.')
        return redirect(url_for('login'))
    return app.send_static_file('batsmen.html')



# Load the Bowler data into a pandas DataFrame
df_bowlers = pd.read_csv('Bowler_data.csv')




@app.route('/search_bowler', methods=['GET'])
def search_bowler():
    if 'username' not in session:
        flash('You need to be logged in to view this page.')
        return redirect(url_for('login'))
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
    if 'username' not in session:
        flash('You need to be logged in to view this page.')
        return redirect(url_for('login'))
    
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)  # Default to 10 items per page
    
    start = (page - 1) * per_page
    end = start + per_page
    
    bowler_data = df_bowlers[start:end].to_dict(orient='records')
    return jsonify(bowler_data)




@app.route('/bowler_data')
def bowler_data():
    if 'username' not in session:
        flash('You need to be logged in to view this page.')
        return redirect(url_for('login'))
    return app.send_static_file('bowler.html')


@app.route('/live_matches', methods=['GET'])
def get_live_matches():
    if 'username' not in session:
        flash('You need to be logged in to view this page.')
        return redirect(url_for('login'))
    
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
    if 'username' not in session:
        flash('You need to be logged in to view this page.')
        return redirect(url_for('login'))
    return app.send_static_file('live.html')


@app.route("/currentMatches", methods=['GET'])
def currentMatches():
    if 'username' not in session:
        flash('You need to be logged in to view this page.')
        return redirect(url_for('login'))
    return app.send_static_file('currentMatches.html')

API_URL = "https://api-inference.huggingface.co/models/google/gemma-7b-it"
HEADERS = {"Authorization": "Bearer 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"}

@app.route("/query", methods=["POST"])
def query():
    payload = request.json
    response = requests.post(API_URL, headers=HEADERS, json=payload)
    return jsonify(response.json())
    
@app.route("/chatbot", methods=['GET'])
def chatbot():
    if 'username' not in session:
        flash('You need to be logged in to view this page.')
        return redirect(url_for('login'))
    return app.send_static_file('Chatbot.html')

@app.route("/players_table")
def players_table():
    return app.send_static_file('players_table.html')

@app.route("/fantasyMatches")
def fantasyMatches():
    return app.send_static_file('FantasyMatches.html')


@app.route("/")
def index():
    if 'username' not in session:
        flash('You need to be logged in to view this page.')
        return redirect(url_for('login'))
    return app.send_static_file('index.html')




if __name__ == '__main__':
    app.run(port=4000,debug=True)