# import required packages
from flask import Flask, request, render_template
from matches import *

import numpy as np
from PIL import Image
from io import BytesIO
import json

from azure.core.exceptions import ResourceNotFoundError
from azure.storage.fileshare import ShareFileClient
from matches import get_result

file_name = "base.pkl"

# define Flask API and return matching images
app = Flask(__name__)


@app.route('/<library_id>', methods=['GET', 'POST'])
def hello_world(library_id):
    # check if reference pickle file already exists for the mentioned library_id
    # If not, download the pickle file for the first time
    reference_images_dir = "/index/" + library_id

    reference_images_pickle = reference_images_dir + "/features/" + file_name
    diff_images_pickle = reference_images_dir + "/features/" + "diff.pkl"

    if request.method == 'GET':
        return render_template('index.html', value='hi')

    if request.method == 'POST':
        if 'file' not in request.files:
            return
        
    file = request.files['file'].read()
    output_json = get_result(file, reference_images_pickle, diff_images_pickle)
    
    return json.dumps(output_json)

#if __name__ == '__main__':
    # app.run(debug=True)
    #app.run(host="0.0.0.0", port=5000)
