FROM continuumio/miniconda

ENV BK_VERSION=2.0.1
ENV PY_VERSION=3.8
ENV NUM_PROCS=1
ENV BOKEH_RESOURCES=cdn

RUN conda config --append channels bokeh
RUN conda install --yes --quiet python=${PY_VERSION} pyyaml jinja2 bokeh=${BK_VERSION} numpy numba scipy sympy "nodejs>=8.8" pandas scikit-learn jupyter
RUN conda install --yes --quiet -c conda-forge lmfit
RUN conda clean -ay

EXPOSE 5006
EXPOSE 80

COPY covid-website.py .
COPY data.py .
COPY ContactRate.py .
COPY SISV.py .
COPY SISV_calib.py .

CMD bokeh serve \
    --allow-websocket-origin="*" \
    --num-procs=${NUM_PROCS} covid-website.py --unused-session-lifetime 3600000 --keep-alive 10000 --log-level info
    
    