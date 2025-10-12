FROM python:3.10-slim

COPY ./services/backend/requirements.txt /tmp/requirements.app.txt
COPY ./libs/requirements.txt /tmp/requirements.libs.txt
RUN pip install --no-cache-dir --upgrade -r /tmp/requirements.app.txt
RUN pip install --no-cache-dir --upgrade -r /tmp/requirements.libs.txt

WORKDIR /src

# Install SSH
RUN apt update && apt install -y make sshpass procps iproute2 iputils-ping dnsutils curl openssh-server
RUN echo 'root:password' | chpasswd && \
    echo 'PasswordAuthentication yes' >> /etc/ssh/sshd_config && \
    echo 'PermitRootLogin yes' >> /etc/ssh/sshd_config && \
    rm -rf /var/lib/apt/lists/*

