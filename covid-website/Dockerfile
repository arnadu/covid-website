FROM continuumio/miniconda

ENV BK_VERSION=2.3.0
ENV PY_VERSION=3.8
ENV NUM_PROCS=1
ENV BOKEH_RESOURCES=cdn

RUN conda config --append channels bokeh
RUN conda install --yes --quiet python=${PY_VERSION} pyyaml flask jinja2 wtforms bokeh=${BK_VERSION} numpy numba scipy sympy "nodejs>=8.8" pandas scikit-learn matplotlib jupyter
RUN conda install --yes --quiet -c conda-forge lmfit flask-executor
RUN pip install flask-session

RUN conda clean -ay

EXPOSE 5006
EXPOSE 80

COPY app.py .
COPY static/index.html static/index.html
COPY templates/study3.html templates/study3.html

COPY object_dict.py .
COPY data.py .
COPY ContactRate.py .
COPY maxlik.py .
COPY SISV.py .
COPY SISV_calib.py .

CMD python app.py