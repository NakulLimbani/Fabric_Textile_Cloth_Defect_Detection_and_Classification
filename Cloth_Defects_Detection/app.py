from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
import os
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Database Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# User Model for storing user data
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)

# Create the database and tables
with app.app_context():
    db.create_all()

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained model
MODEL_PATH = r'C:\Users\spars\Desktop\Cloth_Defect_Detection\model\new_updated_model_version_2correct.keras'
model = tf.keras.models.load_model(MODEL_PATH)

# Mapping of class indices to defect names
CLASS_NAMES = {
    0: "Non-Defective",
    1: "hole",
    2: "objects",
    3: "oil spot",
    4: "thread error"
}

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to preprocess the image before making predictions
def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((64, 64))  # Resize to model input shape
    img = np.array(img) / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to predict defect type
def model_predict(image_path):
    preprocessed_image = preprocess_image(image_path)
    prediction = model.predict(preprocessed_image)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_label = CLASS_NAMES[predicted_class]

    if predicted_label == "Non-Defective":
        return "Non-Defective", None
    else:
        return "Defective", predicted_label

# Home Route
@app.route('/')
def home():
    if 'email' in session:
        user = User.query.filter_by(email=session['email']).first()
        if user:
            return render_template('home.html', username=user.username, welcome_back=True)
        else:
            session.pop('email', None)
            flash('User not found. Please log in again.')
            return redirect(url_for('login'))
    return render_template('home.html')

# Login Route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        if user and user.password == password:
            session['email'] = email
            flash('Login successful!')
            return redirect(url_for('home'))
        else:
            flash('Invalid email or password.')
            return redirect(url_for('login'))
    return render_template('login.html')

# Sign-Up Route
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        user_exists = User.query.filter_by(email=email).first()
        
        if user_exists:
            flash('Email already exists. Please log in.')
            return redirect(url_for('login'))
        
        new_user = User(username=username, email=email, password=password)
        db.session.add(new_user)
        db.session.commit()
        
        flash('Sign-up successful! Please log in.')
        return redirect(url_for('login'))
    return render_template('signup.html')

# Logout Route
@app.route('/logout')
def logout():
    session.pop('email', None)
    flash('You have been logged out.')
    return redirect(url_for('home'))

# Predict Route
@app.route('/predict', methods=['POST'])
def predict():
    if 'email' not in session:
        flash('Please log in to access this feature.')
        return redirect(url_for('login'))

    if 'image' not in request.files:
        flash('No file part')
        return redirect(url_for('home'))

    file = request.files['image']
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('home'))

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Make prediction
        status, defect_type = model_predict(filepath)
        result_message = f"{status} - {defect_type}" if defect_type else status

        return render_template('home.html', filename=filename, result=result_message)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(url_for('home'))

# Contact Us Route
@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')
        flash(f'Thank you for contacting us, {name}! We have received your message.')
        return redirect(url_for('contact'))
    return render_template('contact.html')

# About Us Route
@app.route('/about')
def about():
    return render_template('about.html')

# Feedback Route
@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    if request.method == 'POST':
        f_name = request.form.get('fName')
        f_email = request.form.get('fEmail')
        f_feedback = request.form.get('fFeedback')
        flash(f'Thank you for your feedback, {f_name}! We appreciate your input.')
        return redirect(url_for('feedback'))
    return render_template('feedback.html')

if __name__ == '__main__':
    app.run(debug=True)
