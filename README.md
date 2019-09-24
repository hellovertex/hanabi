# NIP_Hanabi_2019
## Installation (tested on Ubuntu 18.04)
In order to get everything up and running please first run the follwing commands:
```
sudo apt-get install g++       
sudo apt-get install cmake       
sudo apt-get install python-pip
pip3 install -r requirements.txt (RUN THIS COMMAND TWICE FOR NOW)
```

Next compile the environemnt by entering the hanabi_learning_enviroment directory
```
cd hanabi_learning_environment
```

and run:
```
cmake .
make
```
## Training
Training the different algorithms is straight forward. Before running training though, make sure to set the PYTHONPATH environment variable to the root directory of this project. In order to do so, run:
```
export PYTHONPATH="Enter-root-directory-of-this-project-here"
```
You may want to add this line to the .bashrc file in order to run it each time the shell is started.
### Training DQN Agent variations:
Enter the ```training/``` directory and run:
```
python -um train_rainbow_variants --base_dir="ENTER_PREFERRED_BASE_DIR" --gin_files="ENTER_PATH_TO_GIN_FILE_HERE"
```
The ```base_dir path``` specifies the directory where training checkpoints(neural network weights, logs etc.) are saved during training. In general we recommend placing it inside ```agents/trained_models/```.
The ```--gin_files``` flag specifies, which agent you want to train. Each DQN agent has it's own gin file. Find these in ```configs/```. For the specific explanations of the different DQN-agents, please refer to the paper.

### Training with tf-agents: PPO and REINFORCE
Enter the ```training/``` directory and run:
``` 
python3 train_ppo_agent.py --root_dir="ENTER_PREFERRED_BASE_DIR" --alsologtostderr
```
The ```root_dir path``` specifies the directory where training checkpoints(neural network weights, logs etc.) are saved during training.

### Training with rule based agents
To train a rainbow agent in an adhoc setting, we use the gin-config file: ```hanabi_adhoc_rainbow_rule_based```. In it the team number is specified, which determines the number of rainbow agents playing with rule-based agents in a team of 4.
Setting ```create_adhoc_team.team_no``` to ```1``` therefore creates a team with 1 rainbow agent, and 3 rule-based agent. The training is started similar as the rainbow variants:
```
python -um train_adhoc_rainbow_rule_based --base_dir="tmp/adhoc_team1" --gin_files='configs/hanabi_adhoc_rainbow_rule_based.gin'
```



## Evaluate Performances
In order to evaluate how the agents perform in self and adhoc play, run the jupyter-notebook:
```
cd evaluation
jupyter-notebook AdHocViz.ipynb
```
There you see how to run evaluation games with the trained agents and plot their performances.
## Interact with trained agents via Graphical User Interface
### GUI Setup (server on localhost)
These instructions have been tested using Ubuntu 18.04.1 LTS.
```
Install Golang:
	sudo add-apt-repository ppa:longsleep/golang-backports
	(if you don't do this, it will install a version of Golang that is very old)
	sudo apt update
	sudo apt install golang-go -y
	mkdir "$HOME/go"
	export GOPATH=$HOME/go && echo 'export GOPATH=$HOME/go' >> ~/.profile
	export PATH=$PATH:$GOPATH/bin && echo 'export PATH=$PATH:$GOPATH/bin' >> ~/.profile
Install MariaDB:
	sudo apt install mariadb-server -y
	sudo mysql_secure_installation
	Follow the prompts.
Unpack the server to GOPATH:
    unzip the gui/server/go.zip to "$GOPATH/"
Set up a database user and import the database schema:
	sudo mysql -u root -p
	    CREATE DATABASE hanabi;
	    CREATE USER 'hanabiuser'@'localhost' IDENTIFIED BY 'pass';
	    GRANT ALL PRIVILEGES ON hanabi.* to 'hanabiuser'@'localhost';
	    FLUSH PRIVILEGES;
	./install/install_database_schema.sh

COMPILE:
	cd "$HOME/go/src/github.com/Zamiell/hanabi-live/src/"
	go install

RUN SERVER:
	[for some reason you HAVE to start from here !] cd "$HOME/go" 
	sudo "$GOPATH/bin/src" 
	server should run now
	open a browser and type localhost
USE AGENTS:
run the client.py via python3 from within the deepmind repo. 
They will automatically connect to games opened
```

### GUI client
```
Prerequisites:
 - python3
 - running server (See Wiki: GUI-setup-(server-on-localhost) )

Dependencies:
 - python websockets (pip3 install websocket websocket-client)

USAGE:
 - python3 client.py  [defaults to one human player and 2 simple agents]
 - open lobby before or after starting client and agents will join automatically 

EXAMPLEs:
 - python3 client.py -n=0              [n: number of human players --> launches AGENT_ONLY mode]
 - python3 client.py -a simple simple  [a: agent classes as specified in ui_config.py]
 - python3 client.py -e=1              [e: number of episodes agents will play if -n=0]
Also, use -e=1 to finish running games (agents will auto-rejoin), that are hanging due to closed client
 - python3 client.py --help [for more args and info]

Client performs auto-login with created accounts as per default. If this is not desired, delete browsercache or restart server.
```
## Further resources
Please find detailed explanations about the learning environment, encoding of objects, Framework specifics in the wiki of this REPO and theoretical background about this project in the paper.
