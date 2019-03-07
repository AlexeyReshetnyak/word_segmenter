#include <stdint.h>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/core/core.hpp>
#include <string>

using namespace std;
using namespace cv;

static void increase_rect_height(Rect &rect, int32_t addition)
{
  rect.y -= addition/2;
  rect.height += addition/2;
}

static void increase_rect_width(Rect &rect, int32_t addition)
{
  rect.x -= addition/2;
  rect.width += addition/2;
}

int main()
{
  string file_name = "../images/006_ND_1947-01-08_006.tif";
  Mat input_image = imread(file_name);
  Mat image(input_image.rows, input_image.cols, CV_8UC1);
  int32_t addition = 0;
  add(input_image, addition, image);
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
  for(int32_t i = 0; i < contours.size(); i++) {
    rectangle(image, boundRect[i].tl(), boundRect[i].br(), color, 1);
  }

  Mat img_with_incr_rects(image.rows, image.cols, CV_8UC1);
  gray_img.copyTo(img_with_incr_rects);

  for(int32_t i = 0; i < contours.size(); i++) {
    increase_rect_height(boundRect[i], 16);
    increase_rect_width(boundRect[i], 8);
    rectangle(img_with_incr_rects, boundRect[i].tl(), boundRect[i].br(),
              Scalar(0, 0, 0), -1);
  }
  vector<vector<Point>> column_contours;
  vector<Vec4i> column_hierarchy;
  findContours(img_with_incr_rects, column_contours, column_hierarchy,
      CV_RETR_TREE, CV_CHAIN_APPROX_TC89_KCOS, Point(0, 0));
  Mat segm_columnts_img(image.rows, image.cols, CV_8UC1);
  input_image.copyTo(segm_columnts_img);
  for(int32_t i = 0; i < column_contours.size(); i++) {
    approxPolyDP(column_contours[i], contours_poly[i], 3, true);
    boundRect[i] = boundingRect(contours_poly[i]);
  }
  for(int32_t i = 0; i < column_contours.size(); i++) {
    rectangle(segm_columnts_img, boundRect[i].tl(), boundRect[i].br(), color, 1);
  }

  GaussianBlur(img_with_incr_rects, img_with_incr_rects, Size(51, 51), 0, 0);
  imshow("Source image", image);
  imshow("Image with filled rectangles", img_with_incr_rects);
  imshow("Image with segmented columns", segm_columnts_img);
//  imshow("Gray image: ", gray_img);
//  imshow("Filtered image: ", filtered_img);
//  imshow("Binary image: ", binary_img);
  waitKey(0);

  return 0;
}

