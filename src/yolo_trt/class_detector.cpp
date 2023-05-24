#include "yolo_trt/class_detector.h"

#include "yolo_trt/class_yolo_detector.hpp"

namespace yolo_trt {

///
/// \brief The Detector::Impl class
///
class Detector::Impl {
 public:
  Impl() = default;
  virtual ~Impl() = default;

  virtual bool Init(const Config& config) = 0;
  virtual void Detect(const std::vector<cv::Mat>& mat_image,
                      std::vector<BatchResult>& vec_batch_result) = 0;
  virtual cv::Size GetInputSize() const = 0;
  virtual void setNMSThresh(float m_nms_thresh) = 0;
  virtual void setProbThresh(float m_prob_thresh) = 0;
};

///
/// \brief The YoloDectectorImpl class
///
class YoloDectectorImpl final : public Detector::Impl {
 public:
  virtual bool Init(const Config& config) override {
    m_detector.init(config);
    return true;
  }
  virtual void Detect(const std::vector<cv::Mat>& mat_image,
                      std::vector<BatchResult>& vec_batch_result) override {
    m_detector.detect(mat_image, vec_batch_result);
  }
  virtual cv::Size GetInputSize() const override {
    return m_detector.get_input_size();
  }

  virtual void setNMSThresh(float m_nms_thresh) override {
    m_detector.setNMSThresh(m_nms_thresh);
  }
  virtual void setProbThresh(float m_prob_thresh) override {
    m_detector.setProbThresh(m_prob_thresh);
  }

 private:
  YoloDectector m_detector;
};

///
/// \brief Detector::Detector
///
Detector::Detector() noexcept {}

///
/// \brief Detector::~Detector
///
Detector::~Detector() {
  if (m_impl) delete m_impl;
}

///
/// \brief Detector::Init
/// \param config
///
bool Detector::Init(const Config& config) {
  if (m_impl) delete m_impl;

  // if (config.net_type == ModelType::YOLOV6 ||
  //     config.net_type == ModelType::YOLOV7 ||
  //     config.net_type == ModelType::YOLOV7Mask ||
  //     config.net_type == ModelType::YOLOV8)
  //   m_impl = new YoloONNXImpl();
  // else
  m_impl = new YoloDectectorImpl();

  bool res = m_impl->Init(config);
  assert(res);
  return res;
}

///
/// \brief Detector::Detect
/// \param mat_image
/// \param vec_batch_result
///
void Detector::Detect(const std::vector<cv::Mat>& mat_image,
                      std::vector<BatchResult>& vec_batch_result) {
  m_impl->Detect(mat_image, vec_batch_result);
}

///
/// \brief Detector::GetInputSize
/// \return
///
cv::Size Detector::GetInputSize() const { return m_impl->GetInputSize(); }

void Detector::setNMSThresh(float m_nms_thresh) {
  m_impl->setNMSThresh(m_nms_thresh);
}

void Detector::setProbThresh(float m_prob_thresh) {
  m_impl->setProbThresh(m_prob_thresh);
}

}  // namespace yolo_trt
