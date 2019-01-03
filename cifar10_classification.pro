TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
    cifar10_classification.cpp

DESTDIR = $$PWD/bin


INCLUDEPATH += /usr/local/include \
INCLUDEPATH += /usr/local/include/opencv \
INCLUDEPATH += /usr/local/include/opencv2 \
INCLUDEPATH += /usr/local/include/opencv2/core \
INCLUDEPATH += /usr/local/include/opencv2/dnn \
INCLUDEPATH += /usr/local/include/opencv2/highgui \
INCLUDEPATH += /usr/local/include/opencv2/imgcodecs \
INCLUDEPATH += /usr/local/include/opencv2/imgproc \
INCLUDEPATH += /usr/local/include/opencv2/ml \
INCLUDEPATH += /usr/local/include/opencv2/objdetect \
INCLUDEPATH += /usr/local/include/opencv2/video \
INCLUDEPATH += /usr/local/include/opencv2/videoio \

unix:LIBS += `pkg-config opencv --cflags --libs`

