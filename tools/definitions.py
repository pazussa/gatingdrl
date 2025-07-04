import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # newtas root directory

OUT_DIR = os.path.join(ROOT_DIR, 'out')
# Create the output directory if it doesn't exist
os.makedirs(OUT_DIR, exist_ok=True)

CONFIG_DIR = os.path.join(ROOT_DIR, 'config')

LOG_DIR = os.path.join(OUT_DIR, 'log')
os.makedirs(LOG_DIR, exist_ok=True)



