#ifndef CLASS_YOLOV4_H_
#define CLASS_YOLOV4_H_
#include "yolo.h"

namespace yolo_trt {
class YoloV4 : public Yolo {
 public:
  YoloV4(const NetworkInfo &network_info_, const InferParams &infer_params_);

 private:
  std::vector<BBoxInfo> decodeTensor(const int imageIdx, const int imageH,
                                     const int imageW,
                                     const TensorInfo &tensor) override;
};

}  // namespace yolo_trt

#endif
