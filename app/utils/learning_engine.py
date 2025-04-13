import json
import numpy as np
import os
import time
import datetime
from joblib import dump, load
from sklearn.linear_model import LinearRegression

class LearningEngine:
    def __init__(self, db_manager, text_model, image_model):
        #learning engine that improves emotion models over time
        self.db = db_manager
        self.text_model = text_model
        self.image_model = image_model
        
        #MIN VALIDATIONS NEEDED BEFORE TRIGGERING LEARNING - CHANGE HERE IF DESIRED
        self.min_validations = 5
        
        self.models_dir = os.path.join('data', 'models')
        os.makedirs(self.models_dir, exist_ok=True)
        
        self._load_correction_layers()
    
    def _load_correction_layers(self):
        #Load existing correction layers for both models
        #image correction layer (regression model)
        image_corr_path = os.path.join(self.models_dir, 'image_correction.joblib')
        if os.path.exists(image_corr_path):
            try:
                self.image_correction = load(image_corr_path)
                print("loaded image correction layer")
            except:
                print("Failed to load image correction layer, creating new one")
                self.image_correction = None
        else:
            self.image_correction = None
        
        #text correction weights
        text_corr_path = os.path.join(self.models_dir, 'text_correction.joblib')
        if os.path.exists(text_corr_path):
            try:
                self.text_correction = load(text_corr_path)
                print("loaded text correction weights")
            except:
                print("Failed to load text correction weights, creating new ones")
                self.text_correction = None
        else:
            self.text_correction = None
    
    def should_learn(self):
        #check if we have enough validations to trigger learning
        #get pending validations for both models
        text_validations = self.db.get_pending_validations('text')
        image_validations = self.db.get_pending_validations('image')
        
        #check if we have enough validations
        return (len(text_validations) >= self.min_validations or 
                len(image_validations) >= self.min_validations)
    
    def learn(self):
        #Improve models based on collected validations.
        #learning for text emotion model
        text_validations = self.db.get_pending_validations('text')
        if len(text_validations) >= self.min_validations:
            self._learn_text_model(text_validations)
        
        #learning for image emotion model
        image_validations = self.db.get_pending_validations('image')
        if len(image_validations) >= self.min_validations:
            self._learn_image_model(image_validations)
    
    def _learn_text_model(self, validations):
        #Learn from text validations to improve text emotion model
        print(f"Learning from {len(validations)} text validations")
        
        #process validations to create training data
        X = []  #model predictions
        y = []  #user validations
        
        for validation in validations:
            model_emotions = json.loads(validation['emotion_data'])
            user_emotions = json.loads(validation['validated_emotions'])
            
            #convert to feature vectors
            #gna extract all emotion values in a fixed order
            emotions = list(model_emotions.keys())
            
            x_vec = [model_emotions.get(emotion, 0) for emotion in emotions]
            y_vec = [user_emotions.get(emotion, 0) for emotion in emotions]
            
            X.append(x_vec)
            y.append(y_vec)
        
        X = np.array(X)
        y = np.array(y)
        
        #create correction layer (linear regression for simplicity)
        correction = LinearRegression()
        correction.fit(X, y)
        
        # save
        dump(correction, os.path.join(self.models_dir, 'text_correction.joblib'))
        self.text_correction = correction
        
        #update text model to use correction
        self.text_model.set_correction_layer(correction)
        
        #calc accuracy improvement
        accuracy_before = self._calculate_agreement(X, y)
        y_pred = correction.predict(X)
        accuracy_after = self._calculate_agreement(y_pred, y)
        
        #save new model version
        new_version = f"text_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.text_model.update_version(new_version)
        self.db.add_model_version('text', new_version, accuracy_after)
        
        print(f"Text model improved: Accuracy {accuracy_before:.2f} -> {accuracy_after:.2f}")
    
    def _learn_image_model(self, validations):
        #learn from image validations to improve image emotion model
        print(f"Learning from {len(validations)} image validations")
        
        #process validations to create training data
        X = []  #model predictions
        y = []  #user validations
        
        for validation in validations:
            try:
                model_emotions = json.loads(validation['emotion_data'])
                user_emotions = json.loads(validation['validated_emotions'])
                
                #extract emotions from model predictions based on structure
                face_emotions = {}
                if isinstance(model_emotions, list) and len(model_emotions) > 0:
                    #for DeepFace results (list of faces)
                    if 'emotion' in model_emotions[0]:
                        face_emotions = model_emotions[0]['emotion']
                    else:
                        face_emotions = model_emotions[0]  # direct emotion mapping
                elif isinstance(model_emotions, dict):
                    #emotions already in dictionary format
                    face_emotions = model_emotions
                
                #skip if we couldn't extract emotions properly
                if not face_emotions:
                    print(f"Skipping validation {validation['id']} - couldn't extract emotions")
                    continue
                    
                #convert to feature vectors using a fixed list of emotions
                emotions = self.image_model.emotions
                
                x_vec = [face_emotions.get(emotion, 0) for emotion in emotions]
                y_vec = [user_emotions.get(emotion, 0) for emotion in emotions]
                
                X.append(x_vec)
                y.append(y_vec)
            
            except Exception as e:
                print(f"Error processing validation {validation['id']}: {e}")
                continue
        
        #if not enough data after filtering, return
        if len(X) < 2:
            print("Not enough valid data points for training after processing")
            return
        
        X = np.array(X)
        y = np.array(y)
        
        # create correction layer (linear regression for simplicity)
        correction = LinearRegression()
        correction.fit(X, y)
        
        dump(correction, os.path.join(self.models_dir, 'image_correction.joblib'))
        self.image_correction = correction
        
        #update image model to use correction
        self.image_model.set_correction_layer(correction)
        
        #calc accuracy improvement
        accuracy_before = self._calculate_agreement(X, y)
        y_pred = correction.predict(X)
        accuracy_after = self._calculate_agreement(y_pred, y)
        
        #save new model version
        new_version = f"image_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.image_model.update_version(new_version)
        self.db.add_model_version('image', new_version, accuracy_after)
        
        print(f"Image model improved: Accuracy {accuracy_before:.2f} -> {accuracy_after:.2f}")
    
    def _calculate_agreement(self, pred, true):
        #Calculate agreement between predictions and ground truth
        #get top emotion for each sample
        pred_top = np.argmax(pred, axis=1)
        true_top = np.argmax(true, axis=1)
        
        #calc accuracy (% of matching top emotions)
        accuracy = np.mean(pred_top == true_top)
        return accuracy
    
    def get_accuracy_data(self):
        #Get accuracy data for visualization dashboard
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        #get model version history with accuracy
        cursor.execute("""
        SELECT model_type, version, created_date, accuracy 
        FROM model_versions 
        ORDER BY created_date ASC
        """)
        
        versions = cursor.fetchall()
        conn.close()
        
        #organize data for visualization
        text_versions = []
        image_versions = []
        
        for v in versions:
            if v['model_type'] == 'text':
                text_versions.append({
                    'date': v['created_date'],
                    'accuracy': v['accuracy'] or 0,
                    'version': v['version']
                })
            elif v['model_type'] == 'image':
                image_versions.append({
                    'date': v['created_date'],
                    'accuracy': v['accuracy'] or 0,
                    'version': v['version']
                })
        
        return {
            'text': text_versions,
            'image': image_versions
        }