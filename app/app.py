from flask import Flask, render_template, request, redirect, url_for, jsonify, flash
import os
import uuid
import json
from datetime import datetime
from werkzeug.utils import secure_filename

from utils.db_manager import DBManager
from utils.learning_engine import LearningEngine

from models.text_emotion_model import TextEmotionModel
from models.image_emotion_model import ImageEmotionModel

app = Flask(__name__)
app.secret_key = "emotion_dataset_creator_secret_key"
app.config['UPLOAD_FOLDER'] = os.path.join('app', 'static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

db = DBManager('data/emotion_data.db')

text_model = TextEmotionModel()
image_model = ImageEmotionModel()

learning_engine = LearningEngine(db, text_model, image_model)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {
    'text': {'txt', 'md'},
    'image': {'jpg', 'jpeg', 'png'}
}

def allowed_file(filename, file_type):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS[file_type]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    from utils.json_utils import json_serialize
    
    content_type = request.form.get('content_type')
    
    if content_type not in ['text', 'image']:
        flash('Invalid content type')
        return redirect(url_for('index'))
    
    if content_type == 'text':
        text_content = request.form.get('text_content')
        if not text_content:
            flash('No text provided')
            return redirect(url_for('index'))
        
        #generate a unique ID for this upload
        unique_id = str(uuid.uuid4())
        
        filename = f"{unique_id}.txt"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(text_content)
        
        emotions = text_model.analyze(text_content)
        
        media_id = db.add_media('text', filepath)
        analysis_id = db.add_analysis(media_id, text_model.version, json_serialize(emotions))
        
        return redirect(url_for('validate', analysis_id=analysis_id))
    
    elif content_type == 'image':
        if 'image_file' not in request.files:
            flash('No image file provided')
            return redirect(url_for('index'))
        
        file = request.files['image_file']
        
        if file.filename == '':
            flash('No image selected')
            return redirect(url_for('index'))
        
        if not allowed_file(file.filename, 'image'):
            flash('Invalid file type. Only JPG and PNG files are allowed.')
            return redirect(url_for('index'))
        
        #generate a unique ID for this upload
        unique_id = str(uuid.uuid4())
        extension = file.filename.rsplit('.', 1)[1].lower()
        filename = f"{unique_id}.{extension}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        file.save(filepath)
        
        print(f"Saved image to {filepath}")
        
        emotions = image_model.analyze(filepath)
        
        print(f"Image analysis complete, results: {type(emotions)}")
        
        media_id = db.add_media('image', filepath)
        analysis_id = db.add_analysis(media_id, image_model.version, json_serialize(emotions))
        
        return redirect(url_for('validate', analysis_id=analysis_id))

@app.route('/validate/<int:analysis_id>')
def validate(analysis_id):
    #get analysis data from database
    analysis = db.get_analysis(analysis_id)
    if not analysis:
        flash('Analysis not found')
        return redirect(url_for('index'))
    
    media = db.get_media(analysis['media_id'])
    
    text_content = None
    if media['type'] == 'text':
        try:
            with open(media['path'], 'r', encoding='utf-8') as f:
                text_content = f.read()
        except Exception as e:
            print(f"Error reading text file: {e}")
            text_content = "Error reading file content."
    
    data = {
        'analysis_id': analysis_id,
        'media_type': media['type'],
        'media_path': os.path.basename(media['path']),
        'text_content': text_content,
        'emotions': json.loads(analysis['emotion_data'])
    }
    
    return render_template('validate.html', data=data)


@app.route('/submit_validation', methods=['POST'])
def submit_validation():
    from utils.json_utils import json_serialize
    
    analysis_id = request.form.get('analysis_id')
    validated_emotions = {}
    
    #get all emotion sliders from the form
    for key, value in request.form.items():
        if key.startswith('emotion_'):
            emotion = key.replace('emotion_', '')
            validated_emotions[emotion] = float(value) / 100  
    
    #save validation to database using custom JSON serialization
    db.add_validation(analysis_id, json_serialize(validated_emotions))
    
    #check if we have enough validations to trigger learning
    if learning_engine.should_learn():
        learning_engine.learn()
        flash('Models have been updated with your feedback!')
    
    return redirect(url_for('dashboard'))

@app.route('/dashboard')
def dashboard():
    #get stats for dashboard
    stats = db.get_statistics()
    
    #get accuracy improvement data
    accuracy_data = learning_engine.get_accuracy_data()
    
    return render_template('dashboard.html', stats=stats, accuracy_data=accuracy_data)

@app.route('/api/stats')
def api_stats():
    # Get updated stats for AJAX calls
    stats = db.get_statistics()
    return jsonify(stats)

if __name__ == '__main__':
    #create tables if they don't exist
    db.create_tables()
    app.run(debug=True)