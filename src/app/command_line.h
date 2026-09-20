#ifndef PANGU_APP_COMMAND_LINE_H_
#define PANGU_APP_COMMAND_LINE_H_

#include <string>
#include <vector>

namespace pangu::app {

struct CommandLine {
  bool check_input = false;
  bool version = false;
  bool list_plugins = false;
  int exit_code = -1;
  std::vector<std::string> parthenon_arguments;
};

CommandLine ParseCommandLine(int argc, char** argv);

} // namespace pangu::app

#endif
