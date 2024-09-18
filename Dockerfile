ARG HW_TAG=latest
FROM aknh9189/principles-of-autonomy:${HW_TAG}

RUN echo "Running with homework tag: ${HW_TAG}"

COPY jupyter_server_config.py /etc/jupyter/

# Expose the Jupyter Lab port
EXPOSE 9000

WORKDIR /work

# Start Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=9000", "--no-browser", "--allow-root"]

