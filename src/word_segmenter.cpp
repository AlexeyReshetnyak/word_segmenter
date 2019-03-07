#include <stdint.h>
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
  GaussianBlur(gray_img, filtered_img, Size(27, 27), 0, 0);
  Mat binary_img(image.rows, image.cols, CV_8UC1);
  threshold(filtered_img, binary_img, 190, 255, THRESH_BINARY_INV);
  vector<vector<Point>> contours;
  vector<Vec4i> hierarchy;
  findContours(binary_img, contours, hierarchy, CV_RETR_EXTERNAL,
               CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
  Scalar color(0, 0, 255);
  vector<vector<Point>> contours_poly(contours.size());
  vector<Rect> boundRect(contours.size());
  for(int32_t i = 0; i < contours.size(); i++) {
    approxPolyDP(contours[i], contours_poly[i], 3, true);
    boundRect[i] = boundingRect(contours_poly[i]);
  }
  for(int32_t i = 0; i < contours.size(); i++ ) {
    rectangle(image, boundRect[i].tl(), boundRect[i].br(), color, 1);
  }
  imshow("Source image: ", image);
//  imshow("Gray image: ", gray_img);
//  imshow("Filtered image: ", filtered_img);
//  imshow("Binary image: ", binary_img);
  waitKey(0);

  return 0;
}

