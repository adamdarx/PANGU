#include <iostream>
#include <string>
#include <vector>

#include "app/command_line.h"

int main() {
  std::vector<std::string> storage{"pangu",          "--check-input", "-i",
                                   "example.in",     "mesh/nx1=8",   "--flag",
                                   "flag-argument"};
  std::vector<char*> arguments;
  for (auto& argument : storage)
    arguments.push_back(argument.data());

  const auto parsed = pangu::app::ParseCommandLine(static_cast<int>(arguments.size()),
                                                    arguments.data());
  const std::vector<std::string> expected{"pangu", "-i", "example.in", "mesh/nx1=8", "--flag",
                                          "flag-argument"};
  if (!parsed.check_input || parsed.exit_code >= 0 || parsed.parthenon_arguments != expected) {
    std::cerr << "command-line passthrough mismatch:";
    for (const auto& argument : parsed.parthenon_arguments)
      std::cerr << " [" << argument << ']';
    std::cerr << '\n';
    return 1;
  }
  std::cout << "command-line contract passed\n";
  std::vector<std::string> plugin_storage{"pangu", "--list-plugins"};
  arguments.clear();
  for (auto& argument : plugin_storage)
    arguments.push_back(argument.data());
  const auto plugin_list = pangu::app::ParseCommandLine(
      static_cast<int>(arguments.size()), arguments.data());
  if (!plugin_list.list_plugins || plugin_list.exit_code >= 0 ||
      plugin_list.parthenon_arguments != std::vector<std::string>{"pangu"}) {
    std::cerr << "--list-plugins parsing mismatch\n";
    return 1;
  }
  return 0;
}
