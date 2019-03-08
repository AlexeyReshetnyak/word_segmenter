#include <stdint.h>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/core/core.hpp>
#include <string>

using namespace std;
using namespace cv;

static void fill_edges(const Mat &src, Mat &dst, int32_t val, int32_t edge_size)
{
  int32_t border_type = BORDER_CONSTANT | BORDER_ISOLATED;
  Scalar value(val, val, val);
  Rect roi(edge_size, edge_size, src.cols - 2*edge_size,
           src.rows - 2*edge_size);
  copyMakeBorder(src(roi), dst, edge_size, edge_size, edge_size, edge_size,
                 border_type, value);
}

class segmentation
{
public:
  Size kernel_size;
  float sig_x;
  float sig_y;
  float area_threshold_val;
  int32_t thresh;
  int32_t thresh_type;
  int32_t max_val;
  int32_t bound_type;
  int32_t chain_type;
  int32_t border_type;
  int32_t border_size;
  vector<Rect> bound_rects;
  vector<vector<Point>> contours;
  vector<Vec4i> hierarchy;
  Mat binary_img;
  Mat gray_img;
  Mat filtered_img;
  Mat blocks_img;
  Mat black_boxes_img;
  Mat img_with_edges;
  bool process_edges;
  bool area_threshold;

  segmentation(const Mat &img);
  segmentation(const Mat &img, const Mat &src_img);
  void segment();
  void clear_vectors();
  void show_result();
  void show_debug();
  void fill_black_boxes();

private:
  Mat in_img;
};


segmentation::segmentation(const Mat &img)
{
  gray_img.create(img.rows, img.cols, CV_8UC1);
  binary_img.create(img.rows, img.cols, CV_8UC1);
  img_with_edges.create(img.rows, img.cols, CV_8UC1);
  filtered_img.create(img.rows, img.cols, CV_8UC1);
  img.copyTo(blocks_img);
  img.copyTo(in_img);
}

segmentation::segmentation(const Mat &img, const Mat &src_img)
{
  gray_img.create(img.rows, img.cols, CV_8UC1);
  binary_img.create(img.rows, img.cols, CV_8UC1);
  img_with_edges.create(img.rows, img.cols, CV_8UC1);
  filtered_img.create(img.rows, img.cols, CV_8UC1);
  img.copyTo(in_img);
  src_img.copyTo(blocks_img);
}

void segmentation::fill_black_boxes()
{
  gray_img.copyTo(black_boxes_img);
  for(int32_t i = 0; i < contours.size(); i++)
    rectangle(black_boxes_img, bound_rects[i].tl(), bound_rects[i].br(),
        Scalar(0, 0, 255), -1);
}

void segmentation::clear_vectors()
{
  for (int32_t i = 0; i < contours.size(); i++)
    contours[i].clear();
  contours.clear();
  bound_rects.clear();
  hierarchy.clear();
}

void segmentation::segment()
{
  if (in_img.channels() != 1)
    cvtColor(in_img, gray_img, COLOR_BGR2GRAY);
  else
    in_img.copyTo(gray_img);
  GaussianBlur(gray_img, filtered_img, kernel_size, 0, 0);
  threshold(filtered_img, binary_img, thresh, max_val, thresh_type);
  if (process_edges) {
    fill_edges(binary_img, img_with_edges, max_val, border_size);
    findContours(img_with_edges, contours, hierarchy, bound_type,
        chain_type, Point(0, 0));
  }
  else
    findContours(binary_img, contours, hierarchy, bound_type,
        chain_type, Point(0, 0));
  vector<vector<Point>> contours_poly(contours.size());
  bound_rects.reserve(contours.size());
  for(int32_t i = 0; i < contours.size(); i++) {
    approxPolyDP(contours[i], contours_poly[i], 3, true);
    bound_rects[i] = boundingRect(contours_poly[i]);
  }

  Scalar color(0, 0, 255);
  float img_s = in_img.rows * in_img.cols;

  const float max_thresh = 0.8;
  for(int32_t i = 0; i < contours.size(); i++) {
    float rect_s = bound_rects[i].height * bound_rects[i].width;
    if (area_threshold) {
      if (rect_s > area_threshold_val * img_s && rect_s < max_thresh * img_s)
        rectangle(blocks_img, bound_rects[i].tl(), bound_rects[i].br(), color, 1);
    }
    else
      rectangle(blocks_img, bound_rects[i].tl(), bound_rects[i].br(), color, 1);
  }
}

void segmentation::show_result()
{
  imshow("Blocks image", blocks_img);
  waitKey(0);
}

void segmentation::show_debug()
{
  imshow("Filtered image", filtered_img);
  imshow("Binary image", binary_img);
  waitKey(0);
}

int main()
{
  string file_name = "../images/006_ND_1947-01-08_006.tif";
  Mat input_image = imread(file_name);
  segmentation words(input_image);

  words.kernel_size = Size(27, 27);
  words.thresh = 190;
  words.max_val = 255;
  words.bound_type = CV_RETR_EXTERNAL;
  words.chain_type = CV_CHAIN_APPROX_SIMPLE;
  words.thresh_type = THRESH_BINARY_INV;
  words.process_edges = false;
  words.area_threshold = false;

  words.segment();
  words.show_result();
  //words.show_debug();

  words.fill_black_boxes();

  segmentation columns(words.black_boxes_img, input_image);

  columns.kernel_size = Size(25, 25);
  columns.sig_x = 70;
  columns.sig_y = 70;
  columns.thresh = 240;
  columns.max_val = 255;
  columns.bound_type = CV_RETR_TREE;
  columns.chain_type = CV_CHAIN_APPROX_TC89_KCOS;
  columns.thresh_type = THRESH_BINARY;
  columns.border_type = BORDER_CONSTANT;
  columns.process_edges = true;
  columns.border_size = 11;
  columns.area_threshold = true;
  columns.area_threshold_val = 0.01;

  columns.segment();
  columns.show_result();
  //columns.show_debug();
  return 0;
}

