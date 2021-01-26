# Running the python analysis from the provided docker file
## Requirements
A working installation of the docker software package and the docker image is required.
### Software
#### Win10 and macOS
For Windows10 and macOS [docker desktop](https://docs.docker.com/desktop/) can be used.
#### Linux 
For most linux distributions docker is provided by the respective package manager.  
### Docker image
Download the image `ssfacomparison.tar` from [Zenodo](https://doi.org/10.5281/zenodo.4302091) and save it to disk.

## Instructions
#### Open Docker Desktop (on macOS or Win10)
Open the Docker Desktop software and let it run in the background.

#### Open the command line
Open *Terminal.app* (macOS) or *PowerShell* (Win10) or a shell (Linux).

#### Load image
To load the image from disk, run  
`docker image load --input path/ssfa_comparison.tar`  
`path` is where the image is located on your disk. On macOS and Win10 instead of specifying the path to /ssfa_comparison.tar, it is possible to simply drag and drop into the Terminal/Command Prompt window.

#### Run container
Run the image with  
`docker run -p 8888:8888 ssfa_comparison`  
The option `-p` allows you to specify the port.

#### Open the web interface
Open the links in the output of the command above with your browser or go to [localhost:8888](http://localhost:8888), where you need to provide the token from the output of the command above.
Adapt the port if you have changed it above.
## Further remarks
#### Stop the container
In order to stop the running container, you can try to press `crtl+c` on macOS and Linux. For Win10 use the stop function in Docker Desktop.

#### Persistence of results
Docker containers are not designed to be persistent. Thus, when you run the image, change something e.g. in the notebooks and stop the container, the changes will be lost. Hence, it is recommended to save your results using the browser or to set up the analysis on your machine directly.
