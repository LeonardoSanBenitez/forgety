##############################
# Global setup
.ONESHELL: # Source: https://stackoverflow.com/a/30590240
.SILENT: # https://stackoverflow.com/a/11015111

include .env
export $(shell sed 's/=.*//' .env)  # TODO: this fails if the vars are quoted; I think comments lead to error too


##############################
# Functions
# If a function may be executed inside `exec_cluster`, then it must be written in a single line
# Otherwise it may be "normal"
# Some debugging for if ssh connection fails:
#   ps aux | grep sshd
#   ss -tlnp | grep :22
define exec_cluster
	# Run a command inside the master/base/login node of the cluster
	# Arguments:
	#  $(1): Command to run; String; Should be one single command (chain with `;` or `&&` if needed); Multiline commands need to be escaped with backsplash
	# Returns:
	#  Output of the command
	if [ -z "${SSH_USER_SHIBBOLETH}" ]; then \
		echo "Error: SSH_USER_SHIBBOLETH is not set or is empty"; \
		exit 1; \
	fi;
	if ! ssh-keygen -F cl.itk.ppke.hu >/dev/null 2>&1; then \
		echo "Jumphost not found in known hosts (probably because it is connecting for the first time), adding it..."; \
		sshpass -p '${SSH_PASSWORD_SHIBBOLETH}' ssh -4 -o StrictHostKeyChecking=no ${SSH_USER_SHIBBOLETH}@users.itk.ppke.hu pwd; \
		sshpass -p '${SSH_PASSWORD_SHIBBOLETH}' ssh -4 -o StrictHostKeyChecking=no -o ProxyCommand="sshpass -p '${SSH_PASSWORD_CLUSTER}' ssh -W %h:%p ${SSH_USER_SHIBBOLETH}@users.itk.ppke.hu" ${SSH_USER_CLUSTER}@cl.itk.ppke.hu pwd; \
	fi;
	sshpass -p '${SSH_PASSWORD_CLUSTER}' ssh -4 -o StrictHostKeyChecking=no -o ProxyCommand="sshpass -p '${SSH_PASSWORD_SHIBBOLETH}' ssh -W %h:%p ${SSH_USER_SHIBBOLETH}@users.itk.ppke.hu" ${SSH_USER_CLUSTER}@cl.itk.ppke.hu $(1)
endef

define build_cluster
	`# Generates the .sif file from the .def file`\
	`# Can be executed in ssh`\
	`# TODO this is behaving weirdly when I need to build... maybe you need to build manually`\
	cd '${GIT_REPOSITORY_NAME}/infra'; \
	if [ ! -f .env ]; then \
		echo -e 'You must configure the secrets with a local .env file. Exiting...'; \
		exit 1; \
	fi; \
	if [ ! -f run_cluster.sif ]; then \
		echo -e 'Building run_cluster.sif...'; \
		apptainer build --disable-cache run_cluster.sif run_cluster.def; \
	else \
		echo -e 'run_cluster.sif already exists, skipping build...'; \
	fi;
endef

define check_gpu
	# Check the number of available GPUs
	# Arguments:
	#  $(1): GPU ID
	# Returns:
	#  Number of processes running on the GPU
    nvidia-smi -i $(1) --query-compute-apps=pid --format=csv,noheader | wc -l
endef

##############################
# Targets for remote slurm cluster, both batch jobs and interactive jobs
# Docs: https://ppke.sharepoint.com/sites/itk-it/SitePages/HPC.aspx
# All executions are done with apptainer, so you can debug it locally if needed, for example:
# apptainer exec run_cluster.sif python ./fusanalysis/users/Leonardo/3.1_train_cluster.py
# Apptainer has a volume-like mount, so filesystem modifications inside the running "container" are sync with the master node filesystem

setup-cluster:
	# TODO: automate the initial clone of the repo and copy of the .env file
	# For now, you can to do this yourself
	# It's something like:
	# ssh -J USER@users.itk.ppke.hu USER@cl.itk.ppke.hu
	# Check if you are in the folder /home/USER
	# if folder GIT_REPOSITORY_NAME does not exist:
	#   git clone repo
	#   (enter password if needed)
	#   create all necessary .env files with the secrets
	# if folder vision-unlearning does not exist:
	#   git clone https://github.com/LeonardoSanBenitez/vision-unlearning
	# Then you have to push/pull code when necessary, which could also be automated
	echo 'NOT IMPLEMENTED...'
	exit 1

update-lib-cluster:
	$(call exec_cluster, "\
		cd 'vision-unlearning'; \
		git pull; \
		git status; \
		git log -1; \
	")

run-batch-cluster:
	$(call exec_cluster, $(build_cluster))
	$(call exec_cluster, "\
		cd '${GIT_REPOSITORY_NAME}/assets/$(UUID)'; \
		sbatch run_batch_cluster.sh \
	")

clean-cluster:
	$(call exec_cluster, "\
		cd '${GIT_REPOSITORY_NAME}/infra'; \
		rm -rf run_cluster.sif; \
		apptainer cache clean -f; \
	")


stop-one-cluster:
	# @param job_id
	# Usage: make stop-one-cluster job_id=<YOUR_JOB_ID>
	$(call exec_cluster, "\
		scancel $(job_id); \
		echo -e 'Job $(job_id) stopped.' \
	")


debug-one-cluster:
	# Usage: make debug-one-cluster job_id=<YOUR_SLURM JOB_ID> UUID=<Your UUID identifier>
	$(call exec_cluster, "\
		cd '${GIT_REPOSITORY_NAME}/assets/$(UUID)'; \
		JOB_ID=$(job_id); \
		echo -e 'Job ID: \$$JOB_ID'; \
		echo -e '\n\\n-------\nJob details using sacct\n-------'; \
		sacct -j \$$JOB_ID; \
		\
		if [ -f outputs/\$${JOB_ID}-train-err ]; then \
			echo -e '\n\n-------\nJob error\n-------'; \
			cat outputs/\$${JOB_ID}-train-err; \
		else \
			echo -e '\n\n-------\nJob error file does not exist\n-------'; \
		fi; \
		\
		if [ -f outputs/\$${JOB_ID}-train-out ]; then \
			echo -e '\n\n-------\nJob output\n-------'; \
			cat outputs/\$${JOB_ID}-train-out; \
		else \
			echo -e '\n\n-------\nJob output file does not exist\n-------'; \
		fi; \
	")


copy-folder:
	# @param UUID
	# Example usage: make copy-folder UUID=a745ed8a-299b-4705-9958-4470ba7432f7
	$(call exec_cluster, "mkdir -p '${GIT_REPOSITORY_NAME}/assets/$(UUID)'")
	sshpass -p '${SSH_PASSWORD_CLUSTER}' scp -r -o StrictHostKeyChecking=no -o ProxyCommand="sshpass -p '${SSH_PASSWORD_SHIBBOLETH}' ssh -W %h:%p -o StrictHostKeyChecking=no ${SSH_USER_SHIBBOLETH}@users.itk.ppke.hu" -o StrictHostKeyChecking=no /requests/$(UUID) ${SSH_USER_CLUSTER}@cl.itk.ppke.hu:/home/${SSH_USER_CLUSTER}/${GIT_REPOSITORY_NAME}/assets


# works but asks cluster password
#	scp -o ProxyCommand="sshpass -p '${SSH_PASSWORD_CLUSTER}' ssh -W %h:%p ${SSH_USER_SHIBBOLETH}@users.itk.ppke.hu" $(FILE_BATCH) $(FILE_PY) ${SSH_USER_CLUSTER}@cl.itk.ppke.hu:/home/${SSH_USER_CLUSTER}/${GIT_REPOSITORY_NAME}/assets/$(UUID)

# TODO:
# - check job queue
# - check GPU usage
# - cancel specific job
# - copy local file to cluster; unzip in the cluster