# Running the python analysis from the docker file

## Download image
Download the image from [Zenodo](#zenodo.org).

## Load image
Go to the directory where the image is located and run `docker image load ssfa_comparison.tar`.

## Run container
Start the notebooks by running `docker run -p 8891:8888 ssfa_comparison`.
Type `localhost:8891` in your browser. 
