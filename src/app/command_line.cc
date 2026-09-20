#include "app/command_line.h"

#include <string>

#include "CLI11.h"

namespace pangu::app {

CommandLine ParseCommandLine(const int argc, char** argv) {
  CommandLine result;
  CLI::App app{"Performance-portable astrophysical fluid dynamics", "pangu"};
  app.allow_extras();
  app.add_flag("--check-input", result.check_input,
               "Validate the input and print the resolved configuration");
  app.add_flag("--pangu-version", result.version,
               "Print the PANGU build configuration and exit");
  app.add_flag("--list-plugins", result.list_plugins,
               "List plugins compiled into this executable and exit");

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError& error) {
    result.exit_code = app.exit(error);
    return result;
  }

  // CLI11 stores extras in their original command-line order. They are passed
  // directly to Parthenon, rather than fed into another CLI11 parser.
  result.parthenon_arguments = app.remaining();
  result.parthenon_arguments.insert(result.parthenon_arguments.begin(), argv[0]);
  return result;
}

} // namespace pangu::app
