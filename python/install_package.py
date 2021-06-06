# Taken from: https://www.kaggle.com/getting-started/65975

# import
import sys
import subprocess

# install package function
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# example
# install("pathfinding")
