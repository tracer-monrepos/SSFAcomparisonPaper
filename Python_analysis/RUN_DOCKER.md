# Running the python analysis from the provided docker file
## Requirements
A working installation of the docker software package and the docker image is required.
### Software
#### Win10 and macOS
For Windows10 and macOS [docker desktop](https://docs.docker.com/desktop/) can be used.
#### Linux 
For most linux distributions docker is provided by the respective package manager.  
### Docker image
Download the image `ssfacomparison.tar` from [Zenodo](https://doi.org/10.5281/zenodo.4302092) and save it to disk.

### Instructions
#### Open the command line
Open *Terminal.app* (macOS) or *Command Prompt* (Win10) or a shell (Linux).

#### Load image
Run `docker image load --input path/ssfa_comparison.tar`. `path` is where the image is located on your disk. On macOS and Win10 instead of specifying the path to /ssfa_comparison.tar, it is possible to simply drag and drop into the Terminal/Command Prompt window.

#### Run container
Run the image with `docker run -p 8888:8888 ssfa_comparison`.
The option `-p` allows you to specify the port.

#### Open the web interface
Open the links in the output of the command above with your browser or go to [localhost:8888](localhost:8888), where you need to provide the token from the output of the command above.
Adopt the port if you have changed it above.
