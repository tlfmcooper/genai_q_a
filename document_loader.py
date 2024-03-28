import os, pickle
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader
from glob import glob

# Use this line of code if you have a local .env file
load_dotenv() 

import os
from datetime import datetime
from mutagen.mp3 import MP3

def get_meta(file_path):
    details = {}
    
    # Get the file name and extension
    file_name, file_ext = os.path.splitext(os.path.basename(file_path))
    details['filename'] = file_name
    
    # Get interviewee name
    
    
    details['interviewee'] = 'Leo' if 'Bayer' in file_path else 'Louden'
    
    # Get the duration of audio files
    if file_ext.lower() == '.mp3':
        try:
            audio = MP3(file_path)
            details['duration'] = audio.info.length
        except Exception as e:
            print(f"Error getting duration for {file_path}: {e}")
    else:
        details['duration'] = None
    
    return details



def get_documents(input_dir="./data"):
    return SimpleDirectoryReader(input_dir=input_dir, file_metadata=get_meta).load_data(num_workers=4)




if __name__ == "__main__":
    documents = get_documents()
    # Open a file in binary write mode
    with open('interview.pkl', 'wb') as file:
    # Dump the object into the file
        pickle.dump(documents, file)