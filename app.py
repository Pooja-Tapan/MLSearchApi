# import required packages
from flask import Flask, request, render_template
from matches import *

import numpy as np
from PIL import Image
from io import BytesIO
import json

from azure.core.exceptions import ResourceNotFoundError
from azure.storage.fileshare import ShareFileClient

# define azure file download function
def download_azure_file(connection_string, share_name, library_id, file_name, reference_images_pickle):
    try:
        # Build the remote path
        source_file_path = library_id + "/" + file_name

        # Create a ShareFileClient from a connection string
        file_client = ShareFileClient.from_connection_string(connection_string, share_name, source_file_path)

        print("Downloading to:", reference_images_pickle)

        # Open a file for writing bytes on the local system
        with open(reference_images_pickle, "wb") as data:
            # Download the file from Azure into a stream
            stream = file_client.download_file()
            # Write the stream to the local file
            data.write(stream.readall())

    except ResourceNotFoundError as ex:
        print("ResourceNotFoundError:", ex.message)


# initialize connection string and other parameters to download reference pickle file
connection_string = "DefaultEndpointsProtocol=https;AccountName=mvlabs;AccountKey=FWWRSow1FFrskddjEmPnl70PcSk7F2Os5thkRninwcstW3AqgZfPxGlp/QHqCzJNRFOHNhBFK9YfkJCCm57WSA==;EndpointSuffix=core.windows.net"
share_name = "indexstore"
file_name = "rn50.pkl"

# define Flask API and return matching images
app = Flask(__name__)


@app.route('/<library_id>', methods=['GET', 'POST'])
def hello_world(library_id):
    # check if reference pickle file already exists for the mentioned library_id
    # If not, download the pickle file for the first time
    reference_images_dir = "/index/" + library_id
    # os.makedirs(reference_images_dir, exist_ok=True)

    reference_images_pickle = reference_images_dir + "/" + file_name

    # if not (os.path.exists(reference_images_pickle)):
        # print("Downloading Reference Pickle File For the First Time")
        # download_azure_file(connection_string, share_name, library_id, file_name, reference_images_pickle)
    # else:
        # print("Reference Pickle File Already Exists")

    if request.method == 'GET':
        return render_template('index.html', value='hi')

    if request.method == 'POST':
        print(request.files)
        if 'file' not in request.files:
            print('file not uploaded')
            return
        
    file = request.files['file'].read()
    output_json = get_result(file, reference_images_pickle)
    
    return json.dumps(output_json)

#if __name__ == '__main__':
    #app.run(debug=True)
    #app.run(host="0.0.0.0", port=5000)
    
