FROM esmat_training
#python:3.6-alpine3.10

# ENV PACKAGES="\
#     libgcc \
#     libstdc++ \
#     tcl \
#     "

# ENV TOOLS="\
#     py-pip \
#     "

# ENV PYTHON_PACKAGES="\
#     flask-restful \
#     werkzeug \
#     tqdm \
#     requests \
#     matplotlib \
#     scikit-learn[alldeps] \
#     pandas \
#     nltk \
#     seaborn \
#     sklearn \
#     xlrd \
#     pickle-mixin \
#     " 

# RUN apk add --no-cache --virtual build-dependencies $PACKAGES \
#     && apk add --virtual build-runtime \
#     build-base python3-dev openblas-dev freetype-dev pkgconfig gfortran \
#     && ln -s /usr/include/locale.h /usr/include/xlocale.h \
#     && apk update \
#     && apk add make automake gcc g++ subversion python3-dev \
#     && apk add --update $TOOLS \
#     && pip3 install --upgrade pip setuptools \
#     && rm -r /root/.cache 

# RUN pip3 install --no-cache-dir numpy Cython

# RUN pip3 install --no-cache-dir scipy==1.3.3

# RUN pip3 install --no-cache-dir $PYTHON_PACKAGES


WORKDIR /usr/src/app
COPY . .
EXPOSE 5000
CMD [ "python3", "main.py" ]
