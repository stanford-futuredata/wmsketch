/*
 * Binary linear classification with the Weight-Median Sketch, Active-Set Weight-Median Sketch, and baseline methods.
 *
 * Takes as input data in LIBSVM format and outputs a list of features with the highest-magnitude weights in the
 * learned classifier.
 */

#include <chrono>
#include <iostream>
#include <fstream>
#include <random>
#include "cxxopts.hpp"
#include "json.hpp"
#include "util.h"
#include "dataset.h"
#include "topk.h"

using namespace wmsketch;
using json = nlohmann::json;

std::tuple<uint64_t, uint32_t, uint32_t>
train(
    TopKFeatures& topk,
    data::SparseDataset& dataset,
    uint32_t iters = 0,
    uint32_t epochs = 1,
    int32_t seed = 1,
    bool sample = false) {
  uint64_t msecs, runtime_ms;

  tic(msecs);
  uint32_t err_count = 0;
  uint32_t count = 0;
  if (sample && iters == 0) {
    iters = dataset.num_examples();
  }

  if (iters == 0) {
    for (int i = 0; i < epochs; i++) {
      for (auto &ex : dataset) {
        bool yhat = topk.update(ex.features, ex.label == 1);
        if (yhat != ex.label) err_count++;
        count++;
      }
    }
  } else {
    dataset.seed(seed);
    for (int t = 0; t < iters; t++) {
      const data::SparseExample &ex = dataset.sample();
      bool yhat = topk.update(ex.features, ex.label == 1);
      if (yhat != ex.label) err_count++;
      count++;
    }
  }

  runtime_ms = toc(msecs);
  return std::make_tuple(runtime_ms, err_count, count);
}

std::tuple<uint64_t, float, float>
test(
    TopKFeatures& topk,
    data::SparseDataset& dataset) {
  uint64_t msecs, runtime_ms;
  tic(msecs);
  uint32_t tp = 0;
  uint32_t fp = 0;
  uint32_t fn = 0;
  for (auto& ex : dataset) {
    auto y = (ex.label == 1);
    bool yhat = topk.predict(ex.features);
    if (y && yhat) tp++;
    if (!y && yhat) fp++;
    if (y && !yhat) fn++;
  }
  runtime_ms = toc(msecs);
  float precision = (tp + fp == 0) ? 1.f : ((float) tp) / (tp + fp);
  float recall = (tp + fn == 0) ? 1.f : ((float) tp) / (tp + fn);
  return std::make_tuple(runtime_ms, precision, recall);
}

int main(int argc, char **argv) {
  cxxopts::Options options("wmsketch_classification");
  options.add_options()
      ("train", "Train file path", cxxopts::value<std::string>())
      ("test", "Test file path", cxxopts::value<std::string>()->default_value(""))
      ("m,method", "Estimation method", cxxopts::value<std::string>()->default_value("activeset_logistic"))
      ("w,log2_width", "Base-2 logarithm of sketch width", cxxopts::value<uint32_t>()->default_value("10"))
      ("d,depth", "Sketch depth", cxxopts::value<uint32_t>()->default_value("1"))
      ("s,seed", "Random seed", cxxopts::value<int32_t>())
      ("e,epochs", "Number of training epochs", cxxopts::value<uint32_t>()->default_value("1"))
      ("T,iters", "Number of steps in each epoch (0 => number of steps equal to size of dataset)", cxxopts::value<uint32_t>()->default_value("0"))
      ("k,topk", "Number of high-magnitude weights to estimate", cxxopts::value<uint32_t>()->default_value("512"))
      ("lr_init", "Initial learning rate", cxxopts::value<float>()->default_value("0.1"))
      ("l2_reg", "L2 regularization parameter", cxxopts::value<float>()->default_value("1e-6"))
      ("count_smooth", "Laplace smoothing to apply to counts for counter-based baselines", cxxopts::value<float>()->default_value("1.0"))
      ("median_update", "Query WM-Sketch for median weight estimates during each update, instead of updating with random projection of input")
      ("consv_update", "Enable conservative update heuristic for Count-Min sketches")
      ("no_bias", "Train without bias term")
      ("pow", "Exponent for probabilistic truncation method (higher power => less likely to accept low-weight features)", cxxopts::value<float>()->default_value("1.0"))
      ("sample", "Enable sampling of training data instead of making a linear pass")
      ("h,help", "Print help");

  try {
    options.parse(argc, argv);
  } catch (cxxopts::OptionException& e) {
    std::cerr << "Error parsing options: " << e.what() << std::endl;
    std::cerr << options.help() << std::endl;
    exit(1);
  }

  if (options.count("help")) {
    std::cout << options.help() << std::endl;
    exit(0);
  }

  if (!options.count("train")) {
    std::cerr << "Error: train file must be specified" << std::endl;
    std::cerr << options.help() << std::endl;
    exit(1);
  }

  std::string method(options["method"].as<std::string>());
  std::string train_path(options["train"].as<std::string>());
  std::string test_path(options["test"].as<std::string>());
  uint32_t log2_width = options["log2_width"].as<uint32_t>();
  uint32_t depth = options["depth"].as<uint32_t>();
  int32_t seed = options.count("seed") ?
                 options["seed"].as<int32_t>() :
                 (int32_t) std::chrono::system_clock::now().time_since_epoch().count();
  uint32_t iters = options["iters"].as<uint32_t>();
  uint32_t epochs = options["epochs"].as<uint32_t>();
  uint32_t k = options["topk"].as<uint32_t>();
  float lr_init = options["lr_init"].as<float>();
  float l2_reg = options["l2_reg"].as<float>();
  float smooth = options["count_smooth"].as<float>();
  float pow = options["pow"].as<float>();
  bool median_update = (options.count("median_update") != 0);
  bool consv_update = (options.count("consv_update") != 0);
  bool no_bias = (options.count("no_bias") != 0);
  bool sample = (options.count("sample") != 0);

  uint64_t msecs, data_load_ms;
  data::SparseDataset train_dataset, test_dataset;

  std::cerr << "Reading training data from " << train_path << std::endl;
  tic(msecs);
  train_dataset = data::read_libsvm(train_path);
  data_load_ms = toc(msecs);
  std::cerr << "Read training data in " << data_load_ms << "ms" << std::endl;

  if (k == 0) {
    k = train_dataset.feature_dim;
  }

  if (!test_path.empty()) {
    std::cerr << "Reading testing data from " << test_path << std::endl;
    tic(msecs);
    test_dataset = data::read_libsvm(test_path);
    data_load_ms = toc(msecs);
    std::cerr << "Read testing data in " << data_load_ms << "ms" << std::endl;
  }

  json params = {
      {"method", method},
      {"train_path", train_path},
      {"test_path", test_path},
      {"log2_width", log2_width},
      {"depth", depth},
      {"sketch_size", depth * (1 << log2_width)},
      {"seed", seed},
      {"iters", iters},
      {"topk", k},
      {"lr_init", lr_init},
      {"l2_reg", l2_reg},
      {"count_smooth", smooth},
      {"median_update", median_update},
      {"consv_update", consv_update},
      {"no_bias", no_bias},
      {"num_examples", train_dataset.num_examples()},
      {"feature_dim", train_dataset.feature_dim},
      {"pow", pow},
      {"sample", sample}
  };

  std::cerr << params.dump(2) << std::endl;
  std::unique_ptr<TopKFeatures> model;
  if (method == "logistic") {
    model = std::unique_ptr<TopKFeatures>(
        new LogisticTopK(
            k,
            train_dataset.feature_dim,
            lr_init,
            l2_reg,
            no_bias));
  } else if (method == "logistic_sketch") {
    model = std::unique_ptr<TopKFeatures>(
        new LogisticSketchTopK(
            k,
            log2_width,
            depth,
            seed + 1,
            lr_init,
            l2_reg,
            median_update));
  } else if (method == "activeset_logistic") {
    model = std::unique_ptr<TopKFeatures>(
        new ActiveSetLogisticTopK(
            k,
            log2_width,
            depth,
            seed + 1,
            lr_init,
            l2_reg));
  } else if (method == "truncated_logistic") {
    model = std::unique_ptr<TopKFeatures>(
        new TruncatedLogisticTopK(k, lr_init, l2_reg));
  } else if (method == "probtruncated_logistic") {
    model = std::unique_ptr<TopKFeatures>(
        new ProbTruncatedLogisticTopK(k, seed, lr_init, l2_reg, pow));
  } else if (method == "countmin_logistic") {
    model = std::unique_ptr<TopKFeatures>(
        new CountMinLogisticTopK(
            k,
            log2_width,
            depth,
            seed + 1,
            lr_init,
            l2_reg,
            consv_update));
  } else if (method == "spacesaving_logistic") {
    model = std::unique_ptr<TopKFeatures>(
        new SpaceSavingLogisticTopK(
            k,
            seed + 1,
            lr_init,
            l2_reg));
  } else {
    std::cerr << "Error: invalid method " << method << std::endl;
    std::cerr << "Options: logistic, logistic_sketch, activeset_logistic, truncated_logistic, "
              << "probtruncated_logistic, countmin_logistic, spacesaving_logistic" << std::endl;
    std::cerr << options.help() << std::endl;
    exit(1);
  }

  json results;
  uint64_t train_ms;
  uint32_t err_count, count;
  std::tie(train_ms, err_count, count) = train(*model, train_dataset, iters, epochs, seed, sample);
  results["train_ms"] = train_ms;
  results["train_err_count"] = err_count;
  results["train_count"] = count;
  results["train_err_rate"] = double(err_count) / count;
  results["bias"] = model->bias();

  uint64_t test_ms;
  float precision, recall;
  auto test_stats = test(*model, test_dataset);
  std::tie(test_ms, precision, recall) = test_stats;
  results["test_ms"] = test_ms;
  results["test_precision"] = precision;
  results["test_recall"] = recall;
  results["test_f1"] = 2. * precision * recall / (precision + recall);

  std::vector<std::pair<uint32_t, float> > pairs;
  std::vector<uint32_t> indices;
  std::vector<float> values;
  model->topk(pairs);
  for (const auto& p : pairs) {
    indices.push_back(p.first);
    values.push_back(p.second);
  }

  results["top_indices"] = indices;
  results["top_weights"] = values;

  json output;
  output["params"] = params;
  output["results"] = results;
  std::cout << output.dump(0) << std::endl;
  return 0;
}
