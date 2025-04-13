import sys
import os
import numpy as np
import random
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.json_utils import convert_numpy_types

class ImageEmotionModel:
    def __init__(self):
        self.version = "image_v1.0"
        self.correction_layer = None
        self.emotions = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
        
        try:
            sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            import image_to_emotions
            self.original_module = image_to_emotions
            self.has_original_module = True
            print("successfully loaded image_to_emotions module")
        except ImportError as e:
            print(f"Warning: couldn't import image_to_emotions module: {e}")
            print("using fallback emotion analysis (random values for testing)")
            self.has_original_module = False
    
    def analyze(self, image_path):
        #Analyze emotions in image and apply correction if available.
        print(f"Analyzing image: {image_path}")
        
        #get base model predictions
        try:
            if self.has_original_module:
                print("Using og DeepFace module")
                analysis_results = self.original_module.analyze_image_emotions(image_path)
                print(f"analysis results type: {type(analysis_results)}")
                
                #i think fixes json issue (?!) converts numpy types to normal python types
                analysis_results = convert_numpy_types(analysis_results)
                
                if not analysis_results:
                    print("no analysis results, using fallback")
                    analysis_results = self._fallback_analyze(image_path)
            else:
                # fallback: generate random emotions for testing
                print("using fallback analysis")
                analysis_results = self._fallback_analyze(image_path)
            
            #apply correction layer if available
            if self.correction_layer is not None and isinstance(analysis_results, list) and len(analysis_results) > 0:
                if 'emotion' in analysis_results[0]:
                    base_predictions = analysis_results[0]['emotion']
                    
                    # percetange -> [0,1]
                    normalized_predictions = {}
                    for emotion in base_predictions:
                        normalized_predictions[emotion] = base_predictions[emotion] / 100.0
                    
                    #convert to feature vector
                    features = np.array([[normalized_predictions.get(emotion, 0) for emotion in self.emotions]])
                    
                    #apply correction
                    corrected = self.correction_layer.predict(features)[0]
                    
                    #convert back to dict since text vers outputs dict
                    corrected_predictions = {}
                    for i, emotion in enumerate(self.emotions):
                        corrected_predictions[emotion] = max(0, min(1, corrected[i]))
                    
                    #back to percentages
                    for emotion in corrected_predictions:
                        corrected_predictions[emotion] *= 100.0
                    
                    #replace emotion data in the og results
                    analysis_results[0]['emotion'] = corrected_predictions
                    
                    #convert numpy types to normal Python types again
                    analysis_results = convert_numpy_types(analysis_results)
            
            return analysis_results
                
        except Exception as e:
            print(f"Error analyzing image: {e}")
            import traceback
            traceback.print_exc()
            #return fallback values
            return self._fallback_analyze(image_path)
    
    def _fallback_analyze(self, image_path):
        #generate fallback face emotion predictions for testing
        try:
            img = Image.open(image_path)
            width, height = img.size
            
            #fake face region (center of the image)
            center_x = width // 2
            center_y = height // 2
            face_width = min(width, height) // 2
            face_height = face_width
            
            region = {
                'x': center_x - face_width // 2,
                'y': center_y - face_height // 2,
                'w': face_width,
                'h': face_height
            }
            
            #generate random emotions
            emotions = {}
            dominant = random.choice(self.emotions)
            
            for emotion in self.emotions:
                if emotion == dominant:
                    emotions[emotion] = random.uniform(60, 90)
                else:
                    emotions[emotion] = random.uniform(0, 20)
            
            #normalize so they sum to 100
            total = sum(emotions.values())
            for emotion in emotions:
                emotions[emotion] = (emotions[emotion] / total) * 100
            
            #make result similar to DeepFace
            result = {
                'region': region,
                'emotion': emotions
            }
            
            return [result]
            
        except Exception as e:
            print(f"Error in fallback analysis: {e}")
            #return minimal fallback
            emotions = {emotion: 0.0 for emotion in self.emotions}
            emotions["neutral"] = 100.0  #default to neutral
            return [{"emotion": emotions}]
    
    def set_correction_layer(self, correction_layer):
        self.correction_layer = correction_layer
    
    def update_version(self, new_version):
        self.version = new_version