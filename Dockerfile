FROM dolfinx/dolfinx:stable

RUN pip install pyvista
RUN apt-get -qq update && apt-get -y install libgl1-mesa-dev xvfb

RUN git clone https://github.com/missionlab/fenitop /opt/fenitop
ENV PYTHONPATH="/opt/fenitop:${PYTHONPATH}"
