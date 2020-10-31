#include "yolo_trt/class_detector.h"

#include "yolo_trt/class_yolo_detector.hpp"

namespace yolo_trt {

class Detector::Impl {
 public:
  Impl() {}

  ~Impl() {}

  YoloDectector _detector;
};

Detector::Detector() { _impl = new Impl(); }

Detector::~Detector() {
  if (_impl) {
    delete _impl;
    _impl = nullptr;
  }
}

void Detector::init(const Config &config) { _impl->_detector.init(config); }

void Detector::detect(const std::vector<cv::Mat> &mat_image,
                      std::vector<BatchResult> &vec_batch_result) {
  _impl->_detector.detect(mat_image, vec_batch_result);
}

void Detector::setNMSThresh(float m_nms_thresh) {
  _impl->_detector.setNMSThresh(m_nms_thresh);
}

void Detector::setProbThresh(float m_prob_thresh) {
  _impl->_detector.setProbThresh(m_prob_thresh);
}

}  // namespace yolo_trt