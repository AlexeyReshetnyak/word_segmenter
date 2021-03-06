#include "segmenter.hpp"

static void fill_edges(const Mat &src, Mat &dst, int32_t val, int32_t edge_size)
{
  int32_t border_type = BORDER_CONSTANT | BORDER_ISOLATED;
  Scalar value(val, val, val);
  Rect roi(edge_size, edge_size, src.cols - 2*edge_size,
           src.rows - 2*edge_size);
  copyMakeBorder(src(roi), dst, edge_size, edge_size, edge_size, edge_size,
                 border_type, value);
}

segmenter::segmenter(const Mat &img)
{
  gray_img.create(img.rows, img.cols, CV_8UC1);
  binary_img.create(img.rows, img.cols, CV_8UC1);
  img_with_edges.create(img.rows, img.cols, CV_8UC1);
  filtered_img.create(img.rows, img.cols, CV_8UC1);
  img.copyTo(blocks_img);
  img.copyTo(in_img);
}

segmenter::segmenter(const Mat &processed_img, const Mat &src_img)
{
  gray_img.create(src_img.rows, src_img.cols, CV_8UC1);
  binary_img.create(src_img.rows, src_img.cols, CV_8UC1);
  img_with_edges.create(src_img.rows, src_img.cols, CV_8UC1);
  filtered_img.create(src_img.rows, src_img.cols, CV_8UC1);
  processed_img.copyTo(in_img);
  src_img.copyTo(blocks_img);
}

void segmenter::fill_black_boxes()
{
  gray_img.copyTo(black_boxes_img);
  for(int32_t i = 0; i < contours.size(); i++) {
    if (area_threshold) {
      if (check_area(bound_rects[i]))
        rectangle(black_boxes_img, bound_rects[i].tl(), bound_rects[i].br(),
            Scalar(0, 0, 0), -1);
    }
    else
      rectangle(black_boxes_img, bound_rects[i].tl(), bound_rects[i].br(),
          Scalar(0, 0, 0), -1);
    }
}

void segmenter::segment()
{
  if (in_img.channels() != 1)
    cvtColor(in_img, gray_img, COLOR_BGR2GRAY);
  else
    in_img.copyTo(gray_img);
  GaussianBlur(gray_img, filtered_img, kernel_size, sig_x, sig_y,
               border_type);
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

  for(int32_t i = 0; i < contours.size(); i++) {
    if (area_threshold) {
      if (check_area(bound_rects[i]))
        rectangle(blocks_img, bound_rects[i].tl(), bound_rects[i].br(), color,
                  1);
    }
    else
      rectangle(blocks_img, bound_rects[i].tl(), bound_rects[i].br(), color,
                1);
  }
}

bool segmenter::check_area(const Rect &rect)
{
  const float max_thresh = 0.8;
  float img_s = in_img.rows * in_img.cols;
  float rect_s = rect.height * rect.width;

  if (rect_s > area_threshold_val * img_s && rect_s < max_thresh * img_s)
    return true;

  return false;
}

void segmenter::show_result()
{
  imshow("Blocks image", blocks_img);
  waitKey(0);
}

void segmenter::show_debug()
{
  imshow("Input image", in_img);
  waitKey(0);
  imshow("Gray image", gray_img);
  waitKey(0);
  imshow("Filtered image", filtered_img);
  waitKey(0);
  imshow("Binary image", binary_img);
  waitKey(0);
  imshow("Image with edges", img_with_edges);
  waitKey(0);
}

int main(int argc, const char** argv)
{
  cv::CommandLineParser parser(argc, argv,
      "{input_file_name|input_file_name|}"
      );
  std::string input_file_name = parser.get<std::string>("input_file_name");
  Mat input_image = imread(input_file_name);
  segmenter words(input_image);

  words.kernel_size = Size(47, 47);
  words.sig_x = 6;
  words.sig_y = 6;
  words.border_type = BORDER_CONSTANT;
  words.thresh = 190;
  words.max_val = 255;
  words.bound_type = RETR_TREE;
  words.chain_type = CHAIN_APPROX_SIMPLE;
  words.thresh_type = THRESH_BINARY;
  words.process_edges = true;
  words.area_threshold = true;
  words.area_threshold_val = 0.0001;
  words.border_size = 4;

  words.segment();
  words.show_result();

  words.fill_black_boxes();
  words.show_debug();

  segmenter columns(words.black_boxes_img, input_image);

  columns.kernel_size = Size(17, 17);
  columns.sig_x = 12;
  columns.sig_y = 12;
  columns.thresh = 250;
  columns.max_val = 255;
  columns.bound_type = RETR_TREE;
  columns.chain_type = CHAIN_APPROX_TC89_KCOS;
  columns.thresh_type = THRESH_BINARY;
  columns.border_type = BORDER_CONSTANT;
  columns.process_edges = true;
  columns.border_size = 9;
  columns.area_threshold = true;
  columns.area_threshold_val = 0.01;

  columns.segment();
  columns.show_result();
  columns.show_debug();

  return 0;
}

