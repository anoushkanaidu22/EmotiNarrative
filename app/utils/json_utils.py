import json
import numpy as np

def convert_numpy_types(obj):
    #convert numpy types to normal python types for JSON serialization
    #recursively processes dictionaries, lists, and numpy types

    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def json_serialize(obj):
    #serialize an object to JSON string, converting numpy types along the way
    return json.dumps(convert_numpy_types(obj))