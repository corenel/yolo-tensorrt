
#include "yolo_trt/yolov5.h"

#include "yolo_trt/decodeTensorCUDA.h"

namespace yolo_trt {

YoloV5::YoloV5(const NetworkInfo& network_info_,
               const InferParams& infer_params_)
    : Yolo(network_info_, infer_params_) {}
std::vector<BBoxInfo> YoloV5::decodeTensor(const int imageIdx, const int imageH,
                                           const int imageW,
                                           const TensorInfo& tensor) {
  float scale_h = 1.f;
  float scale_w = 1.f;
  int xOffset = 0;
  int yOffset = 0;
  calcuate_letterbox_message(imageH, imageW, scale_h, scale_w, xOffset,
                             yOffset);
  std::vector<BBoxInfo> binfo;

  // decode output (x,y,w,h,maxProb,maxIndex)
  float* boxes = decodeTensorCUDA(imageIdx, tensor);

  // Ergodic result
  for (uint32_t y = 0; y < tensor.grid_h; ++y) {
    for (uint32_t x = 0; x < tensor.grid_w; ++x) {
      for (uint32_t b = 0; b < tensor.numBBoxes; ++b) {
        const int bbindex = y * tensor.grid_w + x;
        const float bx = boxes[18 * bbindex + 6 * b + 0];
        const float by = boxes[18 * bbindex + 6 * b + 1];
        const float bw = boxes[18 * bbindex + 6 * b + 2];
        const float bh = boxes[18 * bbindex + 6 * b + 3];
        const float maxProb = boxes[18 * bbindex + 6 * b + 4];
        const int maxIndex = (int)boxes[18 * bbindex + 6 * b + 5];

        if (maxProb > m_ProbThresh)
          add_bbox_proposal(bx, by, bw, bh, tensor.stride_h, tensor.stride_w,
                            scale_h, scale_w, xOffset, yOffset, maxIndex,
                            maxProb, imageW, imageH, binfo);
      }
    }
  }
  return binfo;
}

}  // namespace yolo_trt
