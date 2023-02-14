FROM tensorflow/tensorflow:2.5.0-gpu-jupyter

ENV TZ=US/Eastern

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y --no-install-recommends apt-utils ants

RUN python3 -m pip install -U pip

RUN pip install --no-cache-dir matplotlib nipype numpy nibabel pandas tqdm scipy pytest-shutil scikit-image SimpleITK jupyter_contrib_nbextensions jupyter_nbextensions_configurator 

RUN jupyter contrib nbextension install --system

RUN jupyter nbextensions_configurator enable --system

USER user 
