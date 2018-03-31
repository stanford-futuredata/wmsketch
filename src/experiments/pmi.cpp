/*
 * Streaming pointwise mutual information (PMI) estimation. PMI is a measure of statistical correlation between
 * random variables -- in this case, between pairs of words
 *
 * Takes as input a collection of text files and outputs a list of bigrams with the highest estimated PMIs.
 */

#include <cstdlib>
#include <cstdint>
#include <chrono>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include "cxxopts.hpp"
#include "json.hpp"
#include "util.h"
#include "sgns.h"

using namespace wmsketch;
using json = nlohmann::json;

int main(int argc, char** argv) {
  cxxopts::Options options("wmsketch_pmi");
  options.add_options()
      ("data", "Whitespace-delimited list of paths", cxxopts::value<std::string>())
      ("w,log2_width", "Log2 of sketch width", cxxopts::value<uint32_t>()->default_value("12"))
      ("d,depth", "Sketch depth", cxxopts::value<uint32_t>()->default_value("1"))
      ("neg_samples", "Negative samples", cxxopts::value<uint32_t>()->default_value("5"))
      ("window_size", "Window size", cxxopts::value<uint32_t>()->default_value("5"))
      ("reservoir_size", "Reservoir size", cxxopts::value<uint32_t>()->default_value("4000"))
      ("s,seed", "Random seed", cxxopts::value<int32_t>())
      ("k,topk", "Top-k feature weights", cxxopts::value<uint32_t>()->default_value("1024"))
      ("lr_init", "Initial learning rate", cxxopts::value<float>()->default_value("0.1"))
      ("l2_reg", "L2 regularization parameter", cxxopts::value<float>()->default_value("1e-7"))
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

  if (!options.count("data")) {
    std::cerr << "Error: train file must be specified" << std::endl;
    std::cerr << options.help() << std::endl;
    exit(1);
  }

  auto& data_paths = options["data"].as<std::string>();
  auto log2_width = options["log2_width"].as<uint32_t>();
  auto depth = options["depth"].as<uint32_t>();
  auto seed = options.count("seed") ?
                 options["seed"].as<int32_t>() :
                 (int32_t) std::chrono::system_clock::now().time_since_epoch().count();
  auto k = options["topk"].as<uint32_t>();
  auto neg_samples = options["neg_samples"].as<uint32_t>();
  auto window_size = options["window_size"].as<uint32_t>();
  auto reservoir_size = options["reservoir_size"].as<uint32_t>();
  float lr_init = options["lr_init"].as<float>();
  float l2_reg = options["l2_reg"].as<float>();

  json params = {
      {"data", data_paths},
      {"log2_width", log2_width},
      {"depth", depth},
      {"seed", seed},
      {"topk", k},
      {"neg_samples", neg_samples},
      {"window_size", window_size},
      {"reservoir_size", reservoir_size},
      {"lr_init", lr_init},
      {"l2_reg", l2_reg}
  };

  std::cerr << params.dump(2) << std::endl;

  StreamingSGNS sgns(
      k,
      log2_width,
      depth,
      neg_samples,
      window_size,
      reservoir_size,
      seed,
      lr_init,
      l2_reg);

  json results;
  uint64_t ms, train_ms;
  uint64_t num_tokens = 0;
  tic(ms);

  // Process tokens in each file
  // Each line is treated as a separate sentence
  std::istringstream path_stream(data_paths);
  std::string data_path;
  while (path_stream >> data_path) {
    std::ifstream ifs(data_path);
    std::string line, token;
    while (std::getline(ifs, line)) {
      std::istringstream iss(line);
      while (iss >> token) {
        std::transform(token.begin(), token.end(), token.begin(), ::tolower);
        sgns.update(token);
        num_tokens++;
      }
      sgns.flush();
    }
  }

  results["train_ms"] = toc(ms);
  results["num_tokens"] = num_tokens;

  // Extract pairs with highest PMI estimates
  std::vector<std::pair<StreamingSGNS::StringPair, float>> pairs;
  std::vector<json> tokens;
  std::vector<float> values;
  sgns.topk(pairs);
  for (const auto& p : pairs) {
    if (p.second < 0) continue;  // skip pairs with negative PMI values
    const auto& s = p.first;
    tokens.push_back({s.first, s.second});
    values.push_back(p.second);
  }

  results["tokens"] = tokens;
  results["weights"] = values;

  json output;
  output["params"] = params;
  output["results"] = results;
  std::cout << output.dump(2) << std::endl;
  return 0;
}
