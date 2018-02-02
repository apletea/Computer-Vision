TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt
CONFIG += link_pkgconfig
PKGCONFIG += opencv

HOME = /home/davinci
CAFFE_DIR = $$HOME/work/caffe-ssd

INCLUDEPATH += $$CAFFE_DIR/include/
INCLUDEPATH += $$CAFFE_DIR/build/src/
INCLUDEPATH += /usr/local/cuda-8.0/include/

LIBS += -L$$CAFFE_DIR/build/lib/ -lcaffe
LIBS += -lboost_system -lboost_filesystem -lboost_regex
LIBS += -lglog

SOURCES += main.cpp \
    boundinbox.cpp \
    helper.cpp \
    regressor.cpp

HEADERS += \
    boundinbox.h \
    helper.h \
    regressor.h
