.DEFAULT_GOAL := help

###########################
# HELP
###########################
include *.mk

###########################
# VARIABLES
###########################
PROJECTNAME := skinmap
GIT_BRANCH := $(shell git rev-parse --abbrev-ref HEAD | tr / _)
PROJECT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/)
DOCKER_BUILD_ARGS ?=

COMMA := ,
DASH := -
EMPTY :=
SPACE := $(EMPTY) $(EMPTY)

# check if `netstat` is installed
ifeq (, $(shell which netstat))
$(error "Netstat executable not found, install it with `apt-get install net-tools`")
endif

# Check if Jupyter Port is already use and define an alternative
ifeq ($(origin PORT), undefined)
  PORT_USED = $(shell netstat -tl | grep -E '(tcp|tcp6)' | grep -Eo '8888' | tail -n 1)
  # Will fail if both ports 9999 and 10000 are used, I am sorry for that
  NEXT_TCP_PORT = $(shell netstat -tl | grep -E '(tcp|tcp6)' | grep -Eo '[0-9]{4}' | sort | tail -n 1 | xargs -I '{}' expr {} + 1)
  ifeq ($(PORT_USED), 8888)
    PORT = $(NEXT_TCP_PORT)
  else
    PORT = 8888
  endif
endif

# docker
ifeq ($(origin CONTAINER_NAME), undefined)
  CONTAINER_NAME := default
endif

ifeq ($(origin LOCAL_DATA_DIR), undefined)
  LOCAL_DATA_DIR := $$PWD/data/
endif

ifeq ($(origin GPU_ID), undefined)
  GPU_ID := all
  GPU_NAME := $(GPU_ID)
else
  GPU_NAME = $(subst $(COMMA),$(DASH),$(GPU_ID))
endif

ifeq ("$(GPU)", "false")
  ifeq (, $(shell which nvidia-smi))
    GPU_ARGS := --shm-size 64G --memory 100G --memory-swap 100G
  else
    GPU_ARGS := --gpus '"device="' --shm-size 64G --memory 100G --memory-swap 100G
  endif
  DOCKER_CONTAINER_NAME := --name $(PROJECTNAME)_$(CONTAINER_NAME)
else
  GPU_ARGS := --gpus '"device=$(GPU_ID)"' --shm-size 64G --ipc=host --memory 100G --memory-swap 100G
  DOCKER_CONTAINER_NAME := --name $(PROJECTNAME)_gpu_$(GPU_NAME)_$(CONTAINER_NAME)
endif

# count elements in comma-seperated GPU list
count = $(words $1)$(if $2,$(call count,$(wordlist 2,$(words $1),$1),$2))
GPU_LIST := $(subst $(COMMA),$(SPACE),$(GPU_ID))
ifeq ($(GPU_ID),all)
  NUM_GPUS := $(shell nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
else
  NUM_GPUS := $(call count,$(GPU_LIST))
endif

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    NUM_CORES := $(shell nproc)
else ifeq ($(UNAME_S),Darwin)
    NUM_CORES := $(shell sysctl -n hw.ncpu)
else
	NUM_CORES := 1
endif
# optimal number of threads is #cores/#gpus
NUM_THREADS := $(shell expr $(NUM_CORES) / $(NUM_GPUS))

# Load environment variables from .env file if it exists
ifneq (,$(wildcard .env))
    include .env
    export
    ENV_FILE_ARG := --env-file .env
else
    ENV_FILE_ARG :=
endif

DOCKER_ARGS := -v $$PWD:/workspace/ -v $(LOCAL_DATA_DIR):/data/ -p $(PORT):8888 --rm $(ENV_FILE_ARG)
DOCKER_CMD := docker run $(DOCKER_ARGS) $(GPU_ARGS) $(DOCKER_CONTAINER_NAME) -it $(PROJECTNAME):$(GIT_BRANCH)

###########################
# PROJECT UTILS
###########################
.PHONY: install
install:  ##@Utils install the dependencies for the project
	@python3 -m pip install -r requirements.txt
	@pre-commit install

.PHONY: clean
clean:  ##@Utils clean the project
	@black .
	@find . -name '*.pyc' -delete
	@find . -name '__pycache__' -type d | xargs rm -fr
	@rm -f .DS_Store
	@rm -f .coverage coverage.xml report.xml
	@rm -f -R .pytest_cache
	@rm -f -R .idea
	@rm -f -R tmp/
	@rm -f -R cov_html/

###########################
# DOCKER
###########################
_build:
	@echo "Build image $(GIT_BRANCH)..."
	@docker build $(DOCKER_BUILD_ARGS) -f Dockerfile -t $(PROJECTNAME):$(GIT_BRANCH) .

run_bash: _build  ##@Docker run an interactive bash inside the docker image (default: GPU=true)
	@echo "Running bash with GPU being $(GPU) and GPU_ID $(GPU_ID)"
	$(DOCKER_CMD) /bin/bash; \

start_jupyter: _build  ##@Docker start a jupyter notebook inside the docker image
	@echo "Starting jupyter notebook"
	@-docker rm $(DOCKER_CONTAINER_NAME)
	$(DOCKER_CMD) /bin/bash -c "jupyter notebook --allow-root --ip 0.0.0.0 --port 8888"

###########################
# TRAINING
###########################

# Training parameters - can be overridden from command line
TRAIN_DATA_CSV ?= ./assets/data.csv
TRAIN_EPOCHS ?= 10
TRAIN_BATCH_SIZE ?= 256
TRAIN_MODEL ?= suinleelab/monet
TRAIN_LR ?= 5e-6
TRAIN_EXTRA_ARGS ?=

.PHONY: train_clip
train_clip: _build ##@Training Train CLIP model (automatically uses all available GPUs)
	@echo "==> Training CLIP with $(NUM_GPUS) GPU(s)"
	@echo "    Model: $(TRAIN_MODEL)"
	@echo "    Data: $(TRAIN_DATA_CSV)"
	@echo "    Epochs: $(TRAIN_EPOCHS)"
	@echo "    Batch size: $(TRAIN_BATCH_SIZE)"
	@echo "    Learning rate: $(TRAIN_LR)"
	@if [ $(NUM_GPUS) -gt 1 ]; then \
		echo "    Using distributed training with torchrun"; \
		$(DOCKER_CMD) bash -c "torchrun --nproc_per_node=$(NUM_GPUS) --master_port=29500 src/train_clip.py \
			--data_csv $(TRAIN_DATA_CSV) \
			--model_name $(TRAIN_MODEL) \
			--epochs $(TRAIN_EPOCHS) \
			--batch_size $(TRAIN_BATCH_SIZE) \
			--lr $(TRAIN_LR) \
			--fp16 \
			$(TRAIN_EXTRA_ARGS)"; \
	else \
		echo "    Using single GPU training"; \
		$(DOCKER_CMD) bash -c "python src/train_clip.py \
			--data_csv $(TRAIN_DATA_CSV) \
			--model_name $(TRAIN_MODEL) \
			--epochs $(TRAIN_EPOCHS) \
			--batch_size $(TRAIN_BATCH_SIZE) \
			--lr $(TRAIN_LR) \
			--fp16 \
			$(TRAIN_EXTRA_ARGS)"; \
	fi
