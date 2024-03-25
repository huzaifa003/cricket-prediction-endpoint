from flask import Flask, request, jsonify, send_from_directory
from joblib import load
import numpy as np

from flask import Flask, request, g, redirect, url_for, render_template, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3

import pandas as pd


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


# Load your trained model
# Make sure to replace 'path/to/your/model.joblib' with the actual path to your model file
model = load('random_forest_model.joblib')


@app.route('/predict', methods=['GET'])
def predict_frontend():
    return send_from_directory(app.static_folder, 'prediction.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)

    # Extracting the input features from the request
    player = data.get('player')
    opposition = data.get('opposition')
    bf = float(data.get('balls_faced', 0))  # Default to 0 if not provided
    ov = float(data.get('overs', 0))        # Default to 0 if not provided

    # Assuming 'test_X' is prepared and available. You might need to adjust this part
    # based on how your model was trained and how your data needs to be prepared.
    # This example assumes that the input features for the prediction are only 'bf' and 'ov'
    # and that the model expects a 2D array-like structure with these features.

    # Prepare the features for prediction
    features = np.array([[1, 1, bf, ov]])  # Assuming the model expects a 2D array-like structure

    # Make prediction
    preds = model.predict(features)
    predicted_runs = preds.astype(int).tolist()

    # Return the prediction result
    return jsonify({
        'player': player,
        'opposition': opposition,
        'predicted_runs': predicted_runs,
        'message': f"{player}'s overall run predicted is {predicted_runs} Against {opposition}"
    })


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


if __name__ == '__main__':
    app.run(port=5000,debug=True)
