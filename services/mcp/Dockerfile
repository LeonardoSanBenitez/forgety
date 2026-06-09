FROM python:3.10-slim

COPY ./services/mcp/requirements.txt /tmp/requirements.mcp.txt
COPY ./libs/requirements.txt /tmp/requirements.libs.txt
RUN pip install --no-cache-dir --upgrade -r /tmp/requirements.mcp.txt
RUN pip install --no-cache-dir --upgrade -r /tmp/requirements.libs.txt

WORKDIR /src
