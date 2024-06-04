FROM --platform=linux/amd64 pytorch/pytorch
# Use a 'large' base container to show-case how to load pytorch and use the GPU (when enabled)

# Ensures that Python output to stdout/stderr is not buffered: prevents missing information when terminating
ENV PYTHONUNBUFFERED 1

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

WORKDIR /opt/app

# Copy nnU-Net results folder into resources folder
# The required files for nnUNet inference are:
# resources/nnUNet_results/.../
# |-- plans.json
# |-- dataset.json
# |-- fold_0/
# |---- checkpoint_final.pth
# |-- fold_1/...
# |-- fold_2/...
# |-- fold_3/...
# |-- fold_4/...
COPY --chown=user:user resources /opt/app/resources

COPY --chown=user:user requirements.txt /opt/app/
# You can add any Python dependencies to requirements.txt
RUN python -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt

# Copy the inference script, the postprocessing script and utils to the container
COPY --chown=user:user inference.py /opt/app/
COPY --chown=user:user postprocess_probability_maps.py /opt/app/
COPY --chown=user:user model.py /opt/app/


ENTRYPOINT ["python", "inference.py"]
