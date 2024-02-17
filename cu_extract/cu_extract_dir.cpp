#include <algorithm>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>
#define PICOJSON_USE_INT64
#include "utils/system.h"
#include <picojson.h>
#include <sys/stat.h>

std::string cuobjdump = "cuobjdump";

std::string sys_run(std::string cmd, std::string a1 = "", std::string a2 = "",
                    std::string a3 = "") {
  std::stringstream ss;
  char buff[100];
  cmd += " " + a1;
  cmd += " " + a2;
  cmd += " " + a3;
  FILE *out = popen(cmd.c_str(), "r");
  while (fgets(buff, 100, out))
    ss << buff;
  pclose(out);
  return ss.str();
}

void getPtxFiles(std::string obj, std::vector<std::string> &files) {
  std::string ptx_file_list = sys_run(cuobjdump, "-lptx", obj);
  static std::regex regex(".*: (.*)\\s+");
  std::smatch m;
  while (std::regex_search(ptx_file_list, m, regex)) {
    files.push_back(m[1]);
    ptx_file_list = m.suffix().str();
  }
}

#include <dirent.h>

void getPtxFilesFromDirectory(const std::string& directory, 
                                    std::vector<std::string>& files) {
  DIR* dir;
  struct dirent* entry;
  std::cerr<<"directory: "<<directory<<std::endl;
  if ((dir = opendir(directory.c_str())) != nullptr) {
    while ((entry = readdir(dir)) != nullptr) {
      std::string filename(entry->d_name);
      std::cerr<<"filename: "<<filename<<std::endl;
      if (filename.length() >= 4 && filename.substr(filename.length() - 4) == ".ptx") {
        files.push_back(filename);
      }
    }
    closedir(dir);
  }
}



struct Param {
  Param() : size(0), align(1), filled(false) {}
  int size;
  int align;
  bool filled;
  picojson::value json() {
    picojson::object obj;
    obj["size"] = picojson::value((int64_t)size);
    obj["align"] = picojson::value((int64_t)align);
    return picojson::value(obj);
  }
};

typedef std::vector<Param> Params;

struct Kernel {
  std::string name;
  std::string target;
  Params params;
  picojson::value json() {
    picojson::object obj;
    obj["name"] = picojson::value(name);
    picojson::array json_params;
    for (auto p : params)
      json_params.push_back(p.json());
    obj["params"] = picojson::value(json_params);
    return picojson::value(obj);
  }
};

typedef std::vector<Kernel> Kernels;

void parseKernelParams(std::string params_str, Params &params) {
  //std::cerr<<"Parameters: "<<params_str<<std::endl;
  std::set<std::string> skip = {".param"};
  std::map<std::string, int> size = {
      {".u8", 1},  {".u16", 2}, {".u32", 4}, {".u64", 8},

      {".s8", 1},  {".s16", 2}, {".s32", 4}, {".s64", 8},

      {".b8", 1},  {".b16", 2}, {".b32", 4}, {".b64", 8},

      {".f16", 2}, {".f32", 4}, {".f64", 8}};
  std::stringstream ss(params_str);
  Param param;
  while (ss) {
    std::string p;
    ss >> p;
    //std::cerr<<"p: "<<p<<std::endl;
    if (!ss)
      break;
    if (p[0] == '.' && skip.count(p) == 0) // Its a .* string and not in skip
    {
      param.filled = true;
      if (p == ".align") {
        ss >> param.align;
        continue;
      }
      if (size.count(p))
        param.size = size.at(p);
      else
        throw std::runtime_error(p);
    } else {
      if (skip.count(p)) { // End Of param
        if (param.filled)
          params.push_back(param);
        param = Param();
      } else { // Paramter name, might include array size
        param.filled = true;
        static std::regex array("\\[(\\d+)\\]");
        std::smatch m;
        int array_size = 1;
        while (std::regex_search(p, m, array)) {
          array_size *= atoi(m[1].str().c_str());
          p = m.suffix().str();
        }
        param.size *= array_size;
        if (param.size <= 0)
          throw std::runtime_error("Zero sized parameter");
      }
    }
  }
  if (param.filled) // Have one unpushed
  {
    if (param.size <= 0)
      throw std::runtime_error("Zero sized parameter");
    params.push_back(param);
  }
}

void getKernels(std::string ptx_file, Kernels &kernels, std::string path) {

  //static std::regex kregex("visible ?\\.entry ([\\S]+)\\((.*)\\)");
  static std::regex kregex(".entry ([\\S]+)\\((.*)\\)");
  // Version regex
  static std::regex vregex("\\.target sm_([0-9]+)");

  std::string ptx;
  { // Scope so they get cleaned up fast
    std::ifstream ifs(path+"/"+ptx_file);
    std::stringstream ss;
    ss << ifs.rdbuf();
    ptx = ss.str();
  }
  std::smatch sm;

  std::string target;
  if (!std::regex_search(ptx, sm, vregex))
    throw std::runtime_error(std::string("No .target definition!"));

  target = sm[1];

  // Make it a single line
  for (auto &c : ptx) {
    if (c == '\n')
      c = ' ';
  }
  // Search for .visible and until the next )
  size_t vis = 0;
   while ( ((vis = ptx.find(".entry", vis)) != std::string::npos) ) {

    size_t epar = ptx.find(")", vis) + 1;
    std::string kernel_def = ptx.substr(vis, epar - vis);
    vis = epar;

    if (!std::regex_search(kernel_def, sm, kregex))
      throw std::runtime_error(std::string("Non matching visible symbol! '") +
                               kernel_def + "'");

    Kernel k;
    k.name = sm[1];
    k.target = target;

    std::stringstream params_ss(sm[2].str());
    std::string param;
    while (std::getline(params_ss, param, ','))
      parseKernelParams(param, k.params);
    kernels.push_back(k);
  }
}

typedef std::map<std::string, Kernels> Modules;

/**
 * Create directory dir.
 * Assumes the full dir path exists if, except the final folder.
 * which will be created by this call, if can_exist is true
 * the final folder can exist.
 */
void make_dir(std::string dir, bool can_exist = false) {
  struct stat st = {0};
  if (stat(dir.c_str(), &st) == -1) {
    if (mkdir(dir.c_str(), 0700) == -1)
      throw std::runtime_error(("Output dir '" + dir) +
                               "' could not be created!");
  } else if (!can_exist)
    throw std::runtime_error(("Output dir '" + dir) +
                             "' already exists,refusing to continue!");
}

/**
 * Create path.
 * Will create all folders necessary for the given path to exist.
 */
void make_path(std::string path) {
  size_t pos = 0;
  size_t slash = 0;

  while ((slash = path.find('/', pos)) != std::string::npos) {
    pos = slash + 1;
    std::string sub_path = path.substr(0, pos);
    make_dir(path.substr(0, pos), true);
  }

  make_dir(path);
}

void move_file(std::string src_file, std::string dst_folder) {
  std::string dst_file = dst_folder + "/" + src_file;
  std::ifstream src(src_file.c_str());
  std::ofstream dst(dst_file.c_str());
  dst << src.rdbuf();
  if (!dst || !src)
    throw std::runtime_error("Rename '" + src_file + "' to '" + dst_file +
                             "' failed");
  src.close();
  dst.close();
  if (unlink(src_file.c_str()))
    throw std::runtime_error("Removing '" + src_file + "' failed");
}

void help(char *argv[]) {
  std::cerr << "Usage:" << std::endl;
  std::cerr << "\t" << argv[0]
            << " <input_dir> <output_dir> "
            << std::endl
            << std::endl;
  std::cerr << "Create a klist with ptx and kernels list from <input dir>, and places "
               "them in folder <output_dir>."
            << std::endl;
  std::cerr << "You can optionaly pass the path to cuobjdump, if it is not in "
               "your PATH.."
            << std::endl;
}

void registerKernelList(std::string new_path) {
  std::string conf_path = system_home_path();
  conf_path += "/.autotalk.json";
  std::ifstream json_in(conf_path);
  picojson::value json;
  picojson::array paths;

  json_in >> json;

  if (json_in)
    paths = json.get<picojson::object>()["paths"].get<picojson::array>();
  else
    json = picojson::value(picojson::object());

  json_in.close();

  for (auto path : paths) {
    std::string old_path = path.get<std::string>();
    if (old_path == new_path)
      return; // Already in the list
  }

  paths.push_back(picojson::value(new_path));

  json.get<picojson::object>()["paths"] = picojson::value(paths);

  std::ofstream json_out(conf_path);
  picojson::value(json).serialize(std::ostream_iterator<char>(json_out), true);
  json_out.close();
}

bool detect_interactive() { return isatty(fileno(stderr)); }

int main(int argc, char *argv[]) {
  if (argc < 3) {
    help(argv);
    return -1;
  }

  bool interactive = detect_interactive();

  // Give so/executable
  std::string path = argv[1]; 
  // "/tmp/venv/lib64/python3.6/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so";
  std::string out = argv[2];

  make_path(out);

  std::vector<std::string> ptx_files;

  getPtxFilesFromDirectory(path, ptx_files);

  std::cerr << "Found " << ptx_files.size() << " modules in " << path
            << std::endl;

  int mid = 1;
  Modules mods;
  for (std::string mod : ptx_files) {
    Kernels k;
    //if (interactive)
    //  std::cerr << "Extracting module " << mod << " " << mid++ << " of "
     //           << ptx_files.size() << " ... ";
    //sys_run(cuobjdump, obj, "-xptx", mod);
    std::cerr<<"ptx: "<<mod<<std::endl;
    getKernels(mod, k, path);
    if (interactive)
      std::cerr << "found " << k.size() << " kernels" << std::endl;
    mods[mod] = k;
  }

  picojson::object json;

  size_t k_cnt = 0;
  for (auto &mod : mods) {
    picojson::array kernels_json;
    std::string target = "";

    //move_file(mod.first, out);

    for (auto &k : mod.second) {
      k_cnt++;
      if (target != "" && target != k.target)
        throw std::runtime_error("Mixed target module '" + mod.first + "'");
      else
        target = k.target;
      kernels_json.push_back(k.json());
    }
    if (json.count(target) == 0)
      json[target] = picojson::value(picojson::object());
    json[target].get<picojson::object>()[mod.first] =
        picojson::value(kernels_json);
  }

  std::ofstream json_file(out + "/klist_pytorch.json");
  picojson::value(json).serialize(std::ostream_iterator<char>(json_file), true);
  json_file.close();
  std::cerr << "Found " << k_cnt << " kernels in " << mods.size() << " modules!"
            << std::endl;


  return 0;
}
