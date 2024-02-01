#include <iostream>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/parsers/sample-models.hpp>
#include <torch/torch.h>

// same as MatrixXf, but with row-major memory layout
//typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXf_rm;

// MatrixXrm<float> x; instead of MatrixXf_rm x;
template <typename V>
using MatrixXrm = typename Eigen::Matrix<V, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

template <typename V>
using VectorXrm = typename Eigen::Vector<V, Eigen::Dynamic>;

// MatrixX<float> x; instead of Eigen::MatrixXf x;
template <typename V>
using MatrixX = typename Eigen::Matrix<V, Eigen::Dynamic, Eigen::Dynamic>;

// VectorX<float> x; instead of Eigen::VectorXf x;
template <typename V>
using VectorX = typename Eigen::Vector<V, Eigen::Dynamic>;

template <typename V>
Eigen::Matrix<V, Eigen::Dynamic, Eigen::Dynamic> libtorch2eigen(torch::Tensor& Tin) {
  /*
   LibTorch is Row-major order and Eigen is Column-major order.
   MatrixXrm uses Eigen::RowMajor for compatibility.
   */
  auto T = Tin.to(torch::kCPU);
  Eigen::Map<MatrixXrm<V>> E(T.data_ptr<V>(), T.size(0), T.size(1));
  return E;
}

template <typename V>
Eigen::Vector<V, Eigen::Dynamic> libtorch2eigenvec(torch::Tensor& Tin) {
  auto T = Tin.to(torch::kCPU);
  Eigen::Map<VectorXrm<V>> E(T.data_ptr<V>(), T.size(0));
  return E;
}

template <typename V>
torch::Tensor eigen2libtorch(MatrixX<V>& M) {
  Eigen::Matrix<V, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> E(M);
  std::vector<int64_t> dims = {E.rows(), E.cols()};
  auto T = torch::from_blob(E.data(), dims).clone(); //.to(torch::kCPU);
  return T;
}

template <typename V>
torch::Tensor eigen2libtorch(VectorX<V>& M) {
  Eigen::Matrix<V, Eigen::Dynamic, 1> E(M);
  std::vector<int64_t> dims = {E.rows()};
  auto T = torch::from_blob(E.data(), dims).clone(); //.to(torch::kCPU);
  return T;
}

int main(int argc, char* argv[]) {
  using namespace pinocchio;

  Model model;
  buildModels::manipulator(model);

  Data data(model);

  // generate random eigen configuration
  Eigen::VectorXd q = randomConfiguration(model);
  Eigen::VectorXf qfloat = q.cast<float>();

  std::cout << "Eigen configuration" << std::endl;
  std::cout << qfloat << std::endl;

  // convert to torch
  auto torch_q = eigen2libtorch(qfloat);
  std::cout << "Libtorch configuration" << std::endl;
  std::cout << torch_q << std::endl;

  auto backward_q = libtorch2eigenvec<float>(torch_q).cast<double>();

  // run with eigen (but inside is torch tensor)
  forwardKinematics(model, data, backward_q);

  Eigen::MatrixXf pos = data.oMi[data.joints.size() - 1].translation().cast<float>();
  std::cout << "Last frame position" << std::endl;
  std::cout << pos << std::endl;

  // convert to torch backward
  auto torch_oMi = eigen2libtorch(pos);
  std::cout << "Libtorch oMi translation" << std::endl;
  std::cout << torch_oMi << std::endl;

  return 0;
}
