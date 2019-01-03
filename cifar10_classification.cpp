#include <iostream>
#include <fstream>
#include <sstream>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace dnn;

std::vector<std::string> classes;

int main(int argc, char** argv)
{
    float scale = 1.0;
    Scalar mean = cv::Scalar(125, 123, 124);
    bool swapRB = true;
    int inpWidth = 32;
    int inpHeight = 32;

    String model = "../models/cifar10_resnet56_iter_64000.caffemodel";
    String config = "../models/cifar10_resnet56_deploy.prototxt";

    String framework = "caffe";

    // Open file with classes names.
    std::string file = "../samples/synset_words.txt";
    std::ifstream ifs(file.c_str());
    if (!ifs.is_open())
        CV_Error(Error::StsError, "File " + file + " not found");
    std::string line;
    while (std::getline(ifs, line))
    {
        classes.push_back(line);
    }

    //! [Read and initialize network]
    Net net = readNet(model, config, framework);
    net.setPreferableBackend(0);
    net.setPreferableTarget(0);
    //! [Read and initialize network]

    // Create a window
    static const std::string kWinName = "Deep learning image classification in OpenCV";
    namedWindow(kWinName, WINDOW_NORMAL);


    std::vector<cv::String> filename_vec;
    cv::String pattern = "../samples/*.jpg";
    cv::glob(pattern, filename_vec, true);

    // Process frames.
    for(int i = 0; i < filename_vec.size(); i++)
    {
        cv::String filename = filename_vec[i];
        Mat frame = cv::imread(filename, cv::IMREAD_COLOR);

        if (frame.empty())
        {
            continue;
        }

//        cv::resize(frame, frame, cv::Size(32,32));

        Mat blob;
        //! [Create a 4D blob from a frame]
        blobFromImage(frame, blob, scale, Size(inpWidth, inpHeight), mean, swapRB, false);
//        blobFromImage(frame, blob, scale, Size(inpWidth, inpHeight));
        //! [Create a 4D blob from a frame]

        //! [Set input blob]
        net.setInput(blob);
        //! [Set input blob]

        //! [Make forward pass]
        Mat prob = net.forward();
        //! [Make forward pass]

        //! [Get a class with a highest score]
        Point classIdPoint;
        double confidence;
        minMaxLoc(prob.reshape(1, 1), 0, &confidence, 0, &classIdPoint);
        int classId = classIdPoint.x;
        //! [Get a class with a highest score]

        // Put efficiency information.
        std::vector<double> layersTimes;
        double freq = getTickFrequency() / 1000;
        double t = net.getPerfProfile(layersTimes) / freq;
        std::string label = format("Inference time: %.2f ms", t);
//        putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
        std::cout << label << std::endl;

        // Print predicted class.
        label = format("%s: %.4f", (classes.empty() ? format("Class #%d", classId).c_str() :
                                                      classes[classId].c_str()),
                                   confidence);
//        putText(frame, label, Point(0, 40), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
        std::cout << label << std::endl;

        imshow(kWinName, frame);
        cv::waitKey(0);
    }
    return 0;
}
