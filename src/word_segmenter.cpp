#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/core/core.hpp>
#include <string>

using namespace std;
using namespace cv;

int main()
{
  string file_name = "../images/006_ND_1947-01-08_006.tif";
  Mat image = imread(file_name);
  Mat gray_img(image.rows, image.cols, CV_8UC1);
  cvtColor(image, gray_img, COLOR_BGR2GRAY);
  Mat filtered_img(image.rows, image.cols, CV_8UC1);
  GaussianBlur(gray_img, filtered_img, Size(41, 41), 0, 0);
  imshow("Source image: ", image);
  imshow("Gray image: ", gray_img);
  imshow("Filtered image: ", filtered_img);
  waitKey(0);

  return 0;
}

