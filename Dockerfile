#######################
# Dockerfile for SSFA #
#######################

# start with python image 
from python:3.8.5

# Make working dir
RUN mkdir /home/SSFA

# Added raw data
ADD R_analysis/derived_data/SSFA_all_data.xlsx /home/SSFA/R_analysis/derived_data/SSFA_all_data.xlsx

# Add analysis data 
ADD Python_analysis/ /home/SSFA/Python_analysis

# Change dir
WORKDIR /home/SSFA

# Install dot
RUN apt-get update && apt-get install -y graphviz

# Upgrade pip
RUN pip3 -q install pip --upgrade

# Install python packages
RUN pip3 install --user -r Python_analysis/requirements.txt

# Start notebook
CMD ["/root/.local/bin/jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
