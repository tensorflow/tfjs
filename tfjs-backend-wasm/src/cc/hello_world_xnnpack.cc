#include <iostream>
#include <xnnpack.h>


int main(int argc, char** argv) {
  const size_t channels = 1, input_stride = 1, output_stride = 1;
  const uint32_t flags = 1;
  xnn_operator_t unary_op = nullptr;

  xnn_status create_status =
      xnn_create_bankers_rounding_nc_f32(
          channels, input_stride, output_stride, flags, &unary_op);
  if (create_status != xnn_status_success) {
    std::cout << "Failed to create op" << std::endl;
    return 1;
  }

  const float* in_buf = new float[1];
  float* out_buf = new float[1];

  xnn_status setup_status =
      xnn_setup_bankers_rounding_nc_f32(
          unary_op, 1, in_buf, out_buf, nullptr /* no thread pool */);

  if (setup_status != xnn_status_success) {
    std::cout << "Failed to setup op" << std::endl;
    return 1;
  }

  xnn_run_operator(unary_op, nullptr /* no threadpool */);
  std::cout << "created xnnpack op" << std::endl;
  return 0;
}
