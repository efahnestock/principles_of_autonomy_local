# Principles of Autonomy Local Development Environment
This repository aims to support running homework for the Principles of Autonomy class locally.


## Mac/Linux Instructions
> Note: If you are running Windows please follow the Windows/WSL instructions below.

### Requirements
You will need to install [Docker Desktop](https://www.docker.com/products/docker-desktop/).

We also highly reccomend installing git to clone and pull from this repo. Follow [the official instructions](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).
Information on the [clone](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository) and [pull](https://www.atlassian.com/git/tutorials/syncing/git-pull) commands.

If you have git installed, you can clone and change directory into this repository using:
```bash
git clone https://github.com/efahnestock/principles_of_autonomy_local.git
cd principles_of_autonomy_local
```
If not, you can download the repository by clicking Code -> Download Zip. This is not reccomended. 


## Instructions for Windows/WSL

*Setup WSL*
- Open Command Prompt  
- ```wsl --install```
- Restart computer if directed 
*Clone repository*
- Open WSL. Do not clone the repository in Windows.
- ```git clone https://github.com/efahnestock/principles_of_autonomy_local.git```
*Setup Docker*
- Follow the steps under "Turn on Docker Desktop WSL 2" at https://docs.docker.com/desktop/wsl/, making sure WSL 2 is clicked during install (it may be selected by default)
- Restart computer if directed
- Open Docker Desktop -> Settings ->Resources -> WSL Integration: Enable integration with additional distros: Ubuntu (change to on). Apply and Restart
*Start Notebook*
- Open WSL
- ```cd principles_of_autonomy_local```
- ```./start_docker.sh ps<homework number>```
- Copy the link and open in browser.

## Downloading the Homework
> Note: On windows/WSL, run all commands inside of WSL.

If you have git, directly pull from this repo to get the newest homework.
```bash
git pull
```
You may run into conflicts you need to [resolve](https://opensource.com/article/23/4/resolve-git-merge-conflicts).

If you do not have git, download the .zip file on Canvas - Modules - Homeworks. For example, for pset 5, the zip file is `ps5-reinforcement-learning.zip`. Place the downloaded folder (you may have to unzip it) in the `homeworks` folder in this repository.

## Running the Homework / Project
First open the docker app. Then, build the docker image by running the following command in the terminal/WSL. Note that your current working directory should be the root level of this repository (`principles_of_autonomy_local/`)
```bash
bash start_docker.sh ps<number>
```
or for projects
```bash
bash start_docker.sh proj<number>
```
where `<number>` is the number of the homework or project you want to run (e.g, `bash start_docker.sh ps2`, `bash start_docker.sh proj1`). Note that if a homework/project hasn't been released yet, you won't be able to run it.

After running the script, you should see a printout of the Jupyter notebook URL. You can copy and paste this URL into your browser to access the Jupyter notebook.

For example, part of the output should look like this: 
```bash
...
jupyterlab-1  | To access the server, open this file in a browser:
jupyterlab-1  |   file:///root/.local/share/jupyter/runtime/jpserver-7-open.html
jupyterlab-1  | Or copy and paste one of these URLs:
jupyterlab-1  |   http://32ab67515b33:9000/lab?token=<some token id>
jupyterlab-1  |   http://127.0.0.1:9000/lab?token=<some token id> <-- USE THIS LINK
...
```
Once you open the last link in your browser, you should see your local files on the left side of your screen. From there, you can open your homework notebook and start working on it.

> **Note:** If you want to use a local editor, most can connect to an existing Jupyter server. You can use the token provided in the output (the 127 one) to connect your editor to the Jupyter server. [Instructions for VSCode](https://code.visualstudio.com/docs/datascience/jupyter-kernel-management#_existing-jupyter-server)

## Stopping the Homework
To stop the homework, simply kill the command running in the terminal by pressing `Ctrl+C`. This will stop the Jupyter notebook server and the Docker container.

## Submitting the Homework
Directly upload all the files in the specific homework folder (e.g., all the files in the folder `homeworks/ps5-reinforcement-learning`) when submitting on Gradescope. Or you can select all files in the homework folder and compress then submit the zip file on Gradescope. Note you shouldn't compress the homework folder directly, instead you need to compress all the files, otherwise there will be path issues on Gradescope causing test failures.

> Note: If you are using WSL follow the first answer [here](https://superuser.com/questions/1324069/how-to-copy-a-file-from-windows-subsystem-for-linux-to-windows-drive-c) to access files in the WSL filesystem.
