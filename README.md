# Principles of Autonomy Local Development Environment
This repository aims to support running homework for the Principles of Autonomy class locally.

## Requirements
You will need to install [Docker Desktop](https://www.docker.com/products/docker-desktop/).

## Running the Homework
First, download the homework from the Deepnote project in the same manner as you do when submitting the homework (check the homework submission guide on Canvas). 

Place the downloaded folder (you may have to unzip it) in the `homeworks` folder in this repository.

Deepnote will place the notebooks from the homework in a `work` folder. **You should move the contents of the `work` folder to the root of the homework folder before running the homework.** E.g., if you download `ps2.zip` from Deepnote, you should unzip it and move the contents of the `ps2/work` folder to `ps2/`. For example `ps2/work/Notebook.ipynb` should be moved to `ps2/Notebook.ipynb`.

Then, run the following command in the terminal. 
```bash start_docker.sh ps<homework number>``` where `<homework number>` is the number of the homework you want to run (e.g, `bash start_docker.sh ps2`). Note that if a homework hasn't been released yet, you won't be able to run it.

> **Note:** If you are using Windows, you may need to run the command in the [Windows Subsystem for Linux](https://learn.microsoft.com/en-us/windows/wsl/) (WSL) terminal, or otherwise run the commands in `start_docker.sh` in a manner that works with your docker setup.

After running the script, you should see a printout of the Jupyter notebook URL. You can copy and paste this URL into your browser to access the Jupyter notebook.

For example, part of the output should look like this: 
```bash
...
jupyterlab-1  |     To access the server, open this file in a browser:
jupyterlab-1  |         file:///root/.local/share/jupyter/runtime/jpserver-7-open.html
jupyterlab-1  |     Or copy and paste one of these URLs:
jupyterlab-1  |         http://32ab67515b33:9000/lab?token=<some token id>
jupyterlab-1  |         http://127.0.0.1:9000/lab?token=<some token id>    <----- USE THIS LINK
...
```
Once you open that link in your browser, you should see your local files on the left side of your screen. From there, you can open your homework notebook and start working on it.

> **Note:** If you want to use a local editor, most can connect to an existing Jupyter server. You can use the token provided in the output to connect your editor to the Jupyter server. [Instructions for VSCode](https://code.visualstudio.com/docs/datascience/jupyter-kernel-management#_existing-jupyter-server)

## Stopping the Homework
To stop the homework, simply kill the command running in the terminal by pressing `Ctrl+C`. This will stop the Jupyter notebook server and the Docker container.
