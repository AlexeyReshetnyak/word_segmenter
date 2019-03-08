#include <stdint.h>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/core/core.hpp>
#include <string>

using namespace cv;
using namespace std;

class segmenter
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

  segmenter(const Mat &img);
  segmenter(const Mat &img, const Mat &src_img);
  void segment();
  void show_result();
  void show_debug();
  void fill_black_boxes();

private:
  Mat in_img;
};


