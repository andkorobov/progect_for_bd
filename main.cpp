#include <iostream>
#include <vector>
#include <exception>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <map>
#include <functional>
#include <sstream>


namespace
{
template <class T>
std::ostream& operator<<(std::ostream& o_out, const std::vector<T>& i_vector)
{
  if (i_vector.empty())
  {
    return o_out << "[]";
  }
  o_out << '[';
  for (size_t i = 0; i + 1 < i_vector.size(); ++i)
  {
    o_out << i_vector[i] << ", ";

  }

  return o_out << i_vector.back() << ']';
}

template <class First, class Second>
std::ostream& operator<<(std::ostream& o_out, const std::pair<First, Second>& i_pair)
{
  return o_out << i_pair.first << ": " << i_pair.second;
}

template <class Key, class Value>
std::ostream& operator<<(std::ostream& o_out, const std::map<Key, Value>& i_map)
{
  if (i_map.empty())
  {
    return o_out << "{}";
  }
  o_out << '{';
  auto end = i_map.end();
  --end;
  for (const auto& pair : i_map)
  {
    o_out << pair;
    if (pair.first != end->first)
    {
      o_out << ", ";
    }
  }

  return o_out << '}';
}

template <class T>
void print(T i_input)
{
  std::cout << i_input << '\n';
}

} //

namespace Math
{
class MathException: public std::exception
{
public:
  MathException(const std::string& i_what);

  virtual const char* what() const throw();
private:
  std::string d_what;
};

using Type = float;
using Vector = std::vector<Type>;
using Matrix = std::vector<Vector>;

double DotProduct(const Vector& i_lhs, const Vector& i_rhs);

Matrix Transpose(const Matrix& i_matrix);

Matrix MatrixProduct(const Matrix& i_lhs, const Matrix& i_rhs);

Matrix Product(const Matrix& i_lhs, const Matrix& i_rhs);

Vector MatrixProduct(const Matrix& i_lhs, const Vector& i_rhs);

Matrix Product(const Matrix& i_lhs, const Vector& i_rhs);

Vector Product(const Vector& i_lhs, const Type& i_rhs);

Type Norm2(const Vector& i_vector);

Type Logaddexp(Type i_x);

Type Expit(Type i_x);

} // Math

namespace LossFunctions
{
enum Type
{
  LOGISTIC,
  QUADRATIC
};
static std::vector<std::string> TypeStr(
{
  "logistic",
  "quadratic"
});

Math::Type Logistic(Math::Type i_M);
Math::Type LogisticGradFactor(Math::Type i_M);
Math::Type Quadratic(Math::Type i_M);
Math::Type QuadraticGradFactor(Math::Type i_M);
Math::Type LossFunctions(Type type, Math::Type i_M);
Math::Type LossFunctionsGradFactor(Type type, Math::Type i_M);
} // namespace LossFunctions

namespace ArgParcer
{
std::string getCmdOption(char ** begin, char ** end, const std::string& option);
bool cmdOptionExists(char** begin, char** end, const std::string& option);

enum LinkType
{
  IDENTITY,
  LOGISTIC,
  GLF1
};

static std::vector<std::string> LinkTypeStr(
{
  "identity",
  "logistic",
  "glf1"
});

struct Inputs
{
  std::string d_inputPath;
  LossFunctions::Type d_lossFunction;
  size_t d_hashSize;
  size_t d_passes;
  Math::Type d_initialT;
  Math::Type d_l1;
  Math::Type d_l2;
  bool d_testonly;
  bool d_adaptive;
  std::string d_finalRegressor;
  std::string d_predictions;
  std::string d_cacheFile;
  LinkType d_link;
  std::vector<std::string> d_quadratic;
};

Inputs GetInputs(int argc, char* argv[]);
} // namespace ArgParcer


void print(const ArgParcer::Inputs& i_inputs)
{
  std::cout << "input_path = " << i_inputs.d_inputPath << '\n'
      << "loss_function = " << LossFunctions::TypeStr[i_inputs.d_lossFunction] << '\n'
      << "hash_size = " << i_inputs.d_hashSize << '\n'
      << "passes = " << i_inputs.d_passes << '\n'
      << "initial_t = " << i_inputs.d_initialT << '\n'
      << "l1 = " << i_inputs.d_l1 << '\n'
      << "l2 = " << i_inputs.d_l2 << '\n'
      << "testonly = " << i_inputs.d_testonly << '\n'
      << "adaptive = " << i_inputs.d_adaptive << '\n';
  if (!i_inputs.d_finalRegressor.empty())
  {
    std::cout << "final_regressor = " << i_inputs.d_finalRegressor << '\n';
  }
  if (!i_inputs.d_cacheFile.empty())
  {
    std::cout << "cache_file = " << i_inputs.d_cacheFile << '\n';
  }
  if (!i_inputs.d_predictions.empty())
  {
    std::cout << "predictions = " << i_inputs.d_predictions << '\n';
  }

  std::cout << "link = " << ArgParcer::LinkTypeStr[i_inputs.d_link] << '\n';

  if (!i_inputs.d_quadratic.empty())
  {
    std::cout << "quadratic = " << i_inputs.d_quadratic << '\n';
  }
}

namespace LileParcer
{
class Hash
{
public:
  Hash();
  size_t operator()(const std::string &s) const;
  void setSize(size_t i_hashSize, size_t i_bithashSize);
  size_t getSize() const
  {
    return d_bithashSize;
  }

private:
  std::hash<std::string> d_hash;
  size_t d_hashSize;
  size_t d_bithashSize;
};
static Hash hash_fn;
struct Feature
{
  size_t d_featureHash;
  Math::Type d_value;
};

std::ostream& operator<<(std::ostream& o_out, const Feature& i_feature);
struct Sample
{
  std::vector<Math::Type> d_lables;
  std::string d_lableName;
  std::map<std::string, std::vector<Feature> > d_nameSpases;
};

std::ostream& operator<<(std::ostream& o_out, const Sample& i_sample);

std::vector<std::string> Split(const std::string& i_line, const std::string& i_sample = " ");

struct NameSpase
{
  std::string d_name;
  std::vector<Feature> d_festures;
};

NameSpase getNameSpase(const std::string& i_line);

Sample GetSampleFromLine(const std::string& i_line, const ArgParcer::Inputs& i_inputs);
} // namespace LileParcer

namespace Learning
{
using Hashes = std::map<std::string, std::vector<Math::Type> >;
using HashesPair = std::pair<std::string, std::vector<Math::Type> >;

Math::Type GetYPredicted(Hashes& io_hashes,
			 const LileParcer::Sample& i_sample,
			 const ArgParcer::Inputs& i_inputs);

void Learning(Hashes& io_hashes,
	      const LileParcer::Sample& i_sample,
	      const Math::Type&  i_lable,
	      const Math::Type&  i_factor,
	      const Math::Type&  i_currentT,
	      const ArgParcer::Inputs& i_inputs,
	      Hashes& io_sqrHashes);

void WriteModel (const ArgParcer::Inputs& i_inputs,
		 const Hashes& i_hashes,
		 const std::string& i_fileName);

void ReadModel (ArgParcer::Inputs& o_inputs,
		Hashes& o_hashes,
		const std::string& i_fileName);

void DoStep(const ArgParcer::Inputs& i_inputs,
	    LileParcer::Sample& io_sample,
	    std::ofstream& o_predictionsFile,
	    Hashes& io_hashes,
	    size_t& io_count,
	    Math::Type& io_lossSum,
	    Math::Type& io_lossSumPrev,
	    size_t& io_prevCount,
	    Hashes& io_sqrHashes);

void PassWithTextFile(const ArgParcer::Inputs& i_inputs,
		     std::ofstream& o_predictionsFile,
		     Hashes& io_hashes,
		     size_t& io_count,
		     Math::Type& io_lossSum,
		     Math::Type& io_lossSumPrev,
		     size_t& io_prevCount,
		     Hashes& io_sqrHashes);

void PassWithCacheFile(const ArgParcer::Inputs& i_inputs,
		       std::ofstream& o_predictionsFile,
		       Hashes& io_hashes,
		       size_t& io_count,
		       Math::Type& io_lossSum,
		       Math::Type& io_lossSumPrev,
		       size_t& io_prevCount,
		       Hashes& io_sqrHashes);

void Run(ArgParcer::Inputs& i_inputs);
} // namespace Learning

//------------------------------------------------------------------------------------------------
//|                                          Main                                                |
//------------------------------------------------------------------------------------------------

int main(int argc, char* argv[])
{
  if (ArgParcer::cmdOptionExists(argv, argv + argc, "--help"))
  {
    std::ifstream infile("help.txt");
    std::string input;
    while (std::getline(infile, input))
    {
      print(input);
    }
    infile.close();
    return 0;
  }
  auto inputs = ArgParcer::GetInputs(argc, argv);
  if (inputs.d_inputPath.empty())
  {
    return 1;
  }
  Learning::Run(inputs);
  return 0;
}

//------------------------------------------------------------------------------------------------
//|                                          Math                                                |
//------------------------------------------------------------------------------------------------

namespace Math
{
MathException::MathException(const std::string& i_what):
  d_what(i_what)
{
}

const char* MathException::what() const throw()
{
  return (std::string("MathException: ") + d_what).c_str();
}

double DotProduct(const Vector& i_lhs, const Vector& i_rhs)
{
  if (i_lhs.size() != i_rhs.size())
  {
    throw MathException("Wrong sizes");
  }
  size_t result = 0;
  for (size_t i = 0; i < i_rhs.size(); ++i)
  {
    result += i_lhs[i] * i_rhs[i];
  }
  return result;
}

Matrix Transpose(const Matrix& i_matrix)
{
  if (i_matrix.empty())
  {
    return i_matrix;
  }
  Matrix
    result(i_matrix[0].size(), Vector(i_matrix.size()));
  for (size_t x = 0; x < i_matrix.size(); ++x)
  {
    for (size_t y = 0; y < i_matrix[0].size(); ++y)
    {
      result[y][x] = i_matrix[x][y];
    }
  }
  return result;
}

Matrix MatrixProduct(const Matrix& i_lhs, const Matrix& i_rhs)
{
  Matrix rhsTranspose = Transpose(i_rhs);
  Matrix result(i_lhs.size(), Vector(rhsTranspose.size()));
  for (size_t x = 0; x < i_lhs.size(); ++x)
  {
    for (size_t y = 0; y < rhsTranspose.size(); ++y)
    {
      result[x][y] = DotProduct(i_lhs[x], rhsTranspose[y]);
    }
  }
  return result;
}

Matrix Product(const Matrix& i_lhs, const Matrix& i_rhs)
{
  Matrix result(i_lhs.size(), Vector(i_lhs[0].size()));
  for (size_t x = 0; x < i_lhs.size(); ++x)
  {
    for (size_t y = 0; y < i_lhs[0].size(); ++y)
    {
      result[x][y] = i_lhs[x][y] * i_rhs[x][y];
    }
  }
  return result;
}

Vector MatrixProduct(const Matrix& i_lhs, const Vector& i_rhs)
{
  Vector result(i_lhs.size());
  for (size_t x = 0; x < i_lhs.size(); ++x)
  {
    result[x] = DotProduct(i_lhs[x], i_rhs);
  }
  return result;
}

Matrix Product(const Matrix& i_lhs, const Vector& i_rhs)
{
  Matrix result(i_lhs.size(), Vector(i_lhs[0].size()));
  for (size_t x = 0; x < i_lhs.size(); ++x)
  {
    for (size_t y = 0; y < i_lhs[0].size(); ++y)
    {
      result[x][y] = i_lhs[x][y] * i_rhs[x];
    }
  }
  return result;
}

Vector Product(const Vector& i_lhs, const Type& i_rhs)
{
  Vector result(i_lhs);
  for (auto& res : result)
  {
      res *= i_rhs;
  }
  return result;
}

Type Norm2(const Vector& i_vector)
{
  Type result = 0;
  for (auto value : i_vector)
  {
    result += value * value;
  }
  return sqrt(result);
}

Type Logaddexp(Type i_x)
{
  if (i_x > 10.)
  {
    return i_x;
  }
  return log(1. + exp(i_x));
}

Type Expit(Type i_x)
{
  if (i_x > 10.)
  {
    return 1.;
  }
  if (i_x < -10.)
  {
    return 0.;
  }
  return 1. / (1. + exp(-i_x));
}
} // Math

//------------------------------------------------------------------------------------------------
//|                                          LossFunctions                                       |
//------------------------------------------------------------------------------------------------

namespace LossFunctions


{
Math::Type Logistic(Math::Type i_M)
{
  return Math::Logaddexp(- i_M);
}

Math::Type LogisticGradFactor(Math::Type i_M)
{
  return - Math::Expit(-i_M);
}

Math::Type Quadratic(Math::Type i_M)
{
  auto los = 1 - i_M;
  return los * los;
}

Math::Type QuadraticGradFactor(Math::Type i_M)
{
  return (i_M - 1) * 2.;
}

Math::Type LossFunctions(Type type, Math::Type i_M)
{
  switch(type)
  {
  case LOGISTIC:
    return Logistic(i_M);
    break;
  case QUADRATIC:
    return Quadratic(i_M);
    break;
  }
  return 0;
}

Math::Type LossFunctionsGradFactor(Type type, Math::Type i_M)
{
  switch(type)
  {
  case LOGISTIC:
    return LogisticGradFactor(i_M);
    break;
  case QUADRATIC:
    return QuadraticGradFactor(i_M);
    break;
  }
  return 0;
}
} // namespace LossFunctions

//------------------------------------------------------------------------------------------------
//|                                          ArgParcer                                           |
//------------------------------------------------------------------------------------------------

namespace ArgParcer
{
std::string getCmdOption(char ** begin, char ** end, const std::string& option)
{
  char ** itr = std::find(begin, end, option);
  if (itr != end && ++itr != end)
  {
    return *itr;
  }
  return "";
}

std::vector<std::string> getCmdOptions(char ** begin, char ** end, const std::string& option)
{
  std::vector<std::string> result;
  while((begin = std::find(begin, end, option)) != end)
  {
    if (++begin != end)
    {
      result.push_back(*begin);
    }
    else
    {
      break;
    }
  }
  return result;
}

bool cmdOptionExists(char** begin, char** end, const std::string& option)
{
    return std::find(begin, end, option) != end;
}

int pow(size_t val, size_t P)
{
  size_t r = 1;
  for (size_t i = 0; i < P; i++)
  {
    r = r * val;
  }
  return r;
}

Inputs GetInputs(int argc, char* argv[])
{
  Inputs inputs;
  if (argc < 2)
  {
    std::cerr << "Usage: " << argv[0] << " input_file_path";
    return inputs;
  }
  inputs.d_inputPath = getCmdOption(argv, argv + argc, "-d");
  if (inputs.d_inputPath.empty())
  {
      inputs.d_inputPath = argv[1];
  }

  inputs.d_lossFunction = LossFunctions::QUADRATIC;
  std::string lossFunctionArg = getCmdOption(argv, argv + argc, "--loss_function");
  if (!lossFunctionArg.empty())
  {
    if (lossFunctionArg == std::string("logistic"))
    {
      inputs.d_lossFunction = LossFunctions::LOGISTIC;
    }
  }

  inputs.d_link = IDENTITY;
  std::string linkArg = getCmdOption(argv, argv + argc, "--link");
  if (!linkArg.empty())
  {
    if (linkArg == std::string("logistic"))
    {
      inputs.d_link = LOGISTIC;
    }
    if (linkArg == std::string("glf1"))
    {
      inputs.d_link = GLF1;
    }
  }

  inputs.d_hashSize = 65536;

  auto bitPrecision = getCmdOption(argv, argv + argc, "--bit_precision");


  if (!bitPrecision.empty())
  {
    size_t intBitPrecision = std::stoi(bitPrecision);
    if (intBitPrecision < 61 && intBitPrecision > 0)
    {
      inputs.d_hashSize = pow(2, intBitPrecision);
      LileParcer::hash_fn.setSize(inputs.d_hashSize, intBitPrecision);
    }
  }

  inputs.d_passes = 1;

  auto strPasses = getCmdOption(argv, argv + argc, "--passes");


  if (!strPasses.empty())
  {
    inputs.d_passes = std::stoi(strPasses);
  }

  inputs.d_initialT = 0.5;

  auto strInitialT = getCmdOption(argv, argv + argc, "--initial_t");


  if (!strInitialT.empty())
  {
    inputs.d_initialT = std::stod(strInitialT);
  }

  inputs.d_l1 = 0.;

  auto strL1 = getCmdOption(argv, argv + argc, "-l1");

  if (!strL1.empty())
  {
    inputs.d_l1 = std::stod(strL1);
  }
  inputs.d_l2 = 0.;

  auto strL2 = getCmdOption(argv, argv + argc, "-l2");

  if (!strL2.empty())
  {
    inputs.d_l2 = std::stod(strL2);
  }

  inputs.d_testonly = cmdOptionExists(argv, argv + argc, "--testonly");
  inputs.d_adaptive = cmdOptionExists(argv, argv + argc, "--adaptive");

  inputs.d_finalRegressor = getCmdOption(argv, argv + argc, "--final_regressor");
  inputs.d_predictions = getCmdOption(argv, argv + argc, "--predictions");

  inputs.d_cacheFile = getCmdOption(argv, argv + argc, "--cache_file");
  if (inputs.d_cacheFile.empty() && cmdOptionExists(argv, argv + argc, "--cache"))
  {
    inputs.d_cacheFile = inputs.d_inputPath + std::string(".cache");
  }

  inputs.d_quadratic = getCmdOptions(argv, argv + argc, "--quadratic");

  return inputs;
}
} // namespace ArgParcer

//------------------------------------------------------------------------------------------------
//|                                          LileParcer                                          |
//------------------------------------------------------------------------------------------------

namespace LileParcer
{
Hash::Hash():
  d_hashSize(65536),
  d_bithashSize(16){}
size_t Hash::operator()(const std::string &s) const
{
  return d_hash(s) % d_hashSize;
}
void Hash::setSize(size_t i_hashSize, size_t i_bithashSize)
{
  d_hashSize = i_hashSize;
  d_bithashSize = i_bithashSize;
}

std::ostream& operator<<(std::ostream& o_out, const Feature& i_feature)
{
  return o_out << '(' << i_feature.d_featureHash << ":" << i_feature.d_value << ')';
}

std::ostream& operator<<(std::ostream& o_out, const Sample& i_sample)
{
  print(i_sample.d_lableName);
  print(i_sample.d_lables);
  print(i_sample.d_nameSpases);
  return o_out;
}

std::vector<std::string> Split(const std::string& i_line, const std::string& i_sample)
{
  std::vector<std::string> result;
  auto sampleStart = i_line.find(i_sample);
  result.push_back(i_line.substr(0, sampleStart));
  if (result.back().empty())
  {
    result.pop_back();
  }
  while(sampleStart < i_line.size())
  {
    sampleStart += i_sample.size();
    auto sampleFinish = i_line.find(i_sample, sampleStart);
    result.push_back(i_line.substr(sampleStart, sampleFinish - sampleStart));
    if (result.back().empty())
    {
      result.pop_back();
    }
    sampleStart = sampleFinish;
  }
  return result;
}

NameSpase getNameSpase(const std::string& i_line)
{
  NameSpase result;
  auto splitedLine = Split(i_line);
  result.d_name = splitedLine[0];

  for (size_t i = 1; i < splitedLine.size(); ++i)
  {
    auto featureNameValue = Split(splitedLine[i], ":");
    if (featureNameValue.size() == 2)
    {
      result.d_festures.push_back({
	  hash_fn(featureNameValue[0]),
	  std::stof(featureNameValue[1])
      });
    }
    else
    {
      result.d_festures.push_back({
	  hash_fn(featureNameValue[0]),
	  1.
      });
    }
  }
  return result;
}

Sample GetSampleFromLine(const std::string& i_line, const ArgParcer::Inputs& i_inputs)
{
  auto splitedLine = Split(i_line, "|");
  Sample result;

  auto lableLineSplited = Split(splitedLine[0]);
  for (auto& lable : lableLineSplited)
  {
    if (lable[0] == '\'')
    {
      result.d_lableName = lable;
      break;
    }
    result.d_lables.push_back(std::stod(lable));
  }
  for (size_t nsCount = 1; nsCount < splitedLine.size(); ++nsCount)
  {
    auto nameSpace = getNameSpase(splitedLine[nsCount]);
    result.d_nameSpases[nameSpace.d_name].swap(nameSpace.d_festures);
  }

  return result;
}
} // namespace LileParcer

//------------------------------------------------------------------------------------------------
//|                                          Learning                                            |
//------------------------------------------------------------------------------------------------

namespace Learning
{
Math::Type GetYPredicted(Hashes& io_hashes,
			 const LileParcer::Sample& i_sample,
			 const ArgParcer::Inputs& i_inputs)
{
  Math::Type yPredicted = 0;
  for (const auto& nameAndNamespace : i_sample.d_nameSpases)
  {
	auto hash = io_hashes.find(nameAndNamespace.first);
	if (hash == io_hashes.end())
	{
	  hash = io_hashes.insert(
	      HashesPair(nameAndNamespace.first,
			 std::vector<Math::Type>
			  (i_inputs.d_hashSize, 0.))).first;
	}
	for (const auto& feature : nameAndNamespace.second)
	{
	    yPredicted += hash->second[feature.d_featureHash] * feature.d_value;
	}
  }
  return yPredicted;
}

void Learning(Hashes& io_hashes,
	      const LileParcer::Sample& i_sample,
	      const Math::Type&  i_lable,
	      const Math::Type&  i_factor,
	      const Math::Type&  i_currentT,
	      const ArgParcer::Inputs& i_inputs,
	      Hashes& io_sqrHashes)
{
  for (const auto& nameAndNamespace : i_sample.d_nameSpases)
  {
    auto hash = io_hashes.find(nameAndNamespace.first);
    auto sqrHashe = io_sqrHashes.find(nameAndNamespace.first);
    if (i_inputs.d_adaptive)
    {
      if (sqrHashe == io_sqrHashes.end())
      {
	sqrHashe = io_sqrHashes.insert(
	    HashesPair(nameAndNamespace.first,
		       std::vector<Math::Type>
			(i_inputs.d_hashSize, 0.))).first;
      }
    }

    for (const auto& feature : nameAndNamespace.second)
    {
      auto& w = hash->second[feature.d_featureHash];
      auto grad = i_lable * i_factor * feature.d_value +
	  w * i_inputs.d_l2 + std::signbit(w)  * i_inputs.d_l1;
      if (grad == 0.)
      {
	continue;
      }
      if (i_inputs.d_adaptive)
      {
	auto& gradSqr = sqrHashe->second[feature.d_featureHash];
	gradSqr += grad * grad;
	grad /= sqrt(gradSqr);
      }
      w -= grad * i_currentT;
    }
  }
}

template <class T>
void SimpleWriteToFile(std::ofstream& o_outfile, const T& i_value)
{
  o_outfile.write((char*)(&i_value), sizeof(T));
}

void SimpleWriteToFile(std::ofstream& o_outfile, const LileParcer::Feature& i_feature)
{
  o_outfile.write((char*)(&i_feature.d_featureHash), (LileParcer::hash_fn.getSize() + 3) / 4);
  SimpleWriteToFile(o_outfile, i_feature.d_value);
}

void WriteStringToFile(std::ofstream& o_outfile, const std::string& i_str)
{
  auto size = i_str.size();
  o_outfile.write(i_str.c_str(), size + 1);
}

template <class T>
void WriteVectorToFile(std::ofstream& o_outfile, const std::vector<T>& i_vector)
{
  auto size = i_vector.size();
  o_outfile.write((char*)(&size), sizeof(size));
  for (const auto& v : i_vector)
  {
    SimpleWriteToFile(o_outfile, v);
  }
}

template <class T>
void WriteNameMapToFile(std::ofstream& o_outfile,
			const std::map<std::string, std::vector<T> >& i_map)
{
  auto size = i_map.size();
  o_outfile.write((char*)(&size), sizeof(size));
  for (const auto& pair : i_map)
  {
    WriteStringToFile(o_outfile, pair.first);
    WriteVectorToFile(o_outfile, pair.second);
  }
}

void WriteModel (const ArgParcer::Inputs& i_inputs,
		 const Hashes& i_hashes,
		 const std::string& i_fileName)
{
  std::ofstream outfile (i_fileName, std::ofstream::binary);
  SimpleWriteToFile(outfile, i_inputs.d_hashSize);
  SimpleWriteToFile(outfile, LileParcer::hash_fn.getSize());
  SimpleWriteToFile(outfile, i_inputs.d_l1);
  SimpleWriteToFile(outfile, i_inputs.d_l2);
  SimpleWriteToFile(outfile, i_inputs.d_lossFunction);
  SimpleWriteToFile(outfile, i_inputs.d_link);

  WriteNameMapToFile(outfile, i_hashes);
  outfile.close();
}

template <class T>
bool ReadSimpleFromFile(std::ifstream& i_infile, T& o_value)
{
  i_infile.read((char*)(&o_value), sizeof(T));
  return !i_infile.eof();
}

bool ReadSimpleFromFile(std::ifstream& i_infile, LileParcer::Feature& o_feature)
{
  o_feature.d_featureHash = 0;
  i_infile.read((char*)(&o_feature.d_featureHash), (LileParcer::hash_fn.getSize() + 3) / 4);

  return ReadSimpleFromFile(i_infile, o_feature.d_value);
}

bool ReadStringFromFile(std::ifstream& i_infile,
			std::string& o_str)
{
  o_str = std::string();
  while (!i_infile.eof())
  {
    char c = 0;
    i_infile.read(&c, 1);
    if (c)
    {
      o_str.push_back(c);
    }
    else
    {
      return !i_infile.eof();
    }
  }

  return false;
}

template <class T>
bool ReadVectorFromFile(std::ifstream& i_infile,
			std::vector<T>& o_vector)
{
  auto size = o_vector.size();
  i_infile.read((char*)(&size), sizeof(size));
  if (i_infile.eof())
  {
    return false;
  }
  for (size_t count = 0; count < size; ++count)
  {
    T value;
    ReadSimpleFromFile(i_infile, value);
    if (i_infile.eof())
    {
      return false;
    }
    o_vector.push_back(value);
  }
  return !i_infile.eof();
}

template <class T>
bool ReadNameMapFromFile(std::ifstream& i_infile,
			 std::map<std::string, std::vector<T> >& o_map)
{
  auto size = o_map.size();
  i_infile.read((char*)(&size), sizeof(size));
  if (i_infile.eof())
  {
    return false;
  }

  for (size_t count = 0; count < size; ++count)
  {
    std::string name;
    if (!ReadStringFromFile(i_infile, name))
    {
      return false;
    }

    auto insertRes =
	o_map.insert(std::pair<std::string, std::vector<T> >
		      (name, std::vector<T>()));

    if (!ReadVectorFromFile(i_infile, insertRes.first->second))
    {
      return false;
    }
  }
  return true;
}

void ReadModel (ArgParcer::Inputs& o_inputs,
		Hashes& o_hashes,
		const std::string& i_fileName)
{
  std::ifstream infile (i_fileName, std::ifstream::binary);
  ReadSimpleFromFile(infile, o_inputs.d_hashSize);
  size_t bitSize;
  ReadSimpleFromFile(infile, bitSize);
  LileParcer::hash_fn.setSize(o_inputs.d_hashSize, bitSize);

  ReadSimpleFromFile(infile, o_inputs.d_l1);
  ReadSimpleFromFile(infile, o_inputs.d_l2);
  ReadSimpleFromFile(infile, o_inputs.d_lossFunction);
  ReadSimpleFromFile(infile, o_inputs.d_link);

  ReadNameMapFromFile(infile, o_hashes);
  infile.close();
}

Math::Type LinkFunction(const ArgParcer::Inputs& i_inputs,
			const Math::Type& i_yPredicted)
{
  switch (i_inputs.d_link)
  {
    case ArgParcer::LOGISTIC:
      return Math::Expit(i_yPredicted);
    case ArgParcer::GLF1:
      return Math::Expit(i_yPredicted) * 2 - 1;
    case ArgParcer::IDENTITY:
      return i_yPredicted;
  }
  return i_yPredicted;
}

void DoStep(const ArgParcer::Inputs& i_inputs,
	    LileParcer::Sample& io_sample,
	    std::ofstream& o_predictionsFile,
	    Hashes& io_hashes,
	    size_t& io_count,
	    Math::Type& io_lossSum,
	    Math::Type& io_lossSumPrev,
	    size_t& io_prevCount,
	    Hashes& io_sqrHashes)
{
  if (i_inputs.d_testonly && io_sample.d_lables.size() != 1)
  {
    io_sample.d_lables = Math::Vector(1, 0.);
  }

  for (const auto& quadratic : i_inputs.d_quadratic)
  {
    auto firstNamespase = io_sample.d_nameSpases.find(quadratic.substr(0, 1));
    auto secondNamespase = io_sample.d_nameSpases.find(quadratic.substr(1, 1));
    if (firstNamespase == io_sample.d_nameSpases.end() ||
	secondNamespase == io_sample.d_nameSpases.end())
    {
      continue;
    }
    LileParcer::NameSpase nameSpace;
    nameSpace.d_name = quadratic;
    for (const auto& firstFeature : firstNamespase->second)
    {
      for (const auto& secondFeature : secondNamespase->second)
      {
	std::stringstream featureName;
	featureName << firstFeature.d_featureHash << ' ' << secondFeature.d_featureHash;
	nameSpace.d_festures.push_back({
	  LileParcer::hash_fn(featureName.str()),
	  1.
	});
      }
    }
    io_sample.d_nameSpases[nameSpace.d_name].swap(nameSpace.d_festures);
  }

  for (const auto& lable : io_sample.d_lables)
  {
    auto yPredicted = GetYPredicted(io_hashes, io_sample, i_inputs);
    auto M = yPredicted * lable;
    Math::Type factor = LossFunctions::LossFunctionsGradFactor(i_inputs.d_lossFunction, M);
    auto currentT = i_inputs.d_initialT / io_count;
    if (i_inputs.d_testonly)
    {
      o_predictionsFile << LinkFunction(i_inputs, yPredicted) << '\n';
    }
    else
    {
      Learning(io_hashes, io_sample, lable, factor, currentT, i_inputs, io_sqrHashes);
    }

    auto loss = LossFunctions::LossFunctions(i_inputs.d_lossFunction, M);
    io_lossSum += loss;
    io_lossSumPrev += loss;

    if (++io_count > io_prevCount * 2)
    {
      io_prevCount *= 2;
      std::cout << io_prevCount;
      if (io_prevCount < 10000000.)
      {
	std::cout << "\t";
      }
      std::cout << "\t| " << std::fixed;
      std::cout.precision(8);
      std::cout << io_lossSum / io_prevCount << "\t| ";
      std::cout << io_lossSumPrev / (io_prevCount / 2) << "\t| ";
      std::cout << lable << "\t| ";
      std::cout << LinkFunction(i_inputs, yPredicted) << "\t| ";
      std::cout << currentT << "\n";
      io_lossSumPrev = 0;
    }
  }
}

void WriteSampleToChach(std::ofstream& o_chachFile,
			const LileParcer::Sample& i_sample)
{
  WriteStringToFile(o_chachFile, i_sample.d_lableName);
  WriteVectorToFile(o_chachFile, i_sample.d_lables);
  WriteNameMapToFile(o_chachFile, i_sample.d_nameSpases);
}

bool ReadSample(std::ifstream& i_chachFile,
		 LileParcer::Sample& o_sample)
{
  o_sample = LileParcer::Sample();
  if(!ReadStringFromFile(i_chachFile, o_sample.d_lableName))
  {
    return false;
  }
  if(!ReadVectorFromFile(i_chachFile, o_sample.d_lables))
  {
    return false;
  }

  if(!ReadNameMapFromFile(i_chachFile, o_sample.d_nameSpases))
  {
    return false;
  }
  return true;
}

void PassWithTextFile(const ArgParcer::Inputs& i_inputs,
		     std::ofstream& o_predictionsFile,
		     Hashes& io_hashes,
		     size_t& io_count,
		     Math::Type& io_lossSum,
		     Math::Type& io_lossSumPrev,
		     size_t& io_prevCount,
		     Hashes& io_sqrHashes)
{
  std::string input;
  std::ifstream inputFile(i_inputs.d_inputPath);

  std::ofstream cacheFile (i_inputs.d_cacheFile, std::ofstream::binary);

  while (std::getline(inputFile, input))
  {
    auto sample = LileParcer::GetSampleFromLine(input, i_inputs);

    WriteSampleToChach(cacheFile, sample);

    DoStep(i_inputs, sample, o_predictionsFile,
	   io_hashes, io_count, io_lossSum, io_lossSumPrev,
	   io_prevCount, io_sqrHashes);
  }
  cacheFile.close();
  inputFile.close();
}

void PassWithCacheFile(const ArgParcer::Inputs& i_inputs,
		       std::ofstream& o_predictionsFile,
		       Hashes& io_hashes,
		       size_t& io_count,
		       Math::Type& io_lossSum,
		       Math::Type& io_lossSumPrev,
		       size_t& io_prevCount,
		       Hashes& io_sqrHashes)
{
  std::ifstream inputFile(i_inputs.d_cacheFile);
  LileParcer::Sample sample;
  while (ReadSample(inputFile, sample))
  {
    DoStep(i_inputs, sample, o_predictionsFile,
	   io_hashes, io_count, io_lossSum, io_lossSumPrev,
	   io_prevCount, io_sqrHashes);
  }
  inputFile.close();
}

void Run(ArgParcer::Inputs& i_inputs)
{
  size_t count = 1;
  size_t prevCount = 1;
  Hashes hashes;
  Hashes sqrHashes; // for adaptive

  if (i_inputs.d_testonly)
  {
    ReadModel(i_inputs, hashes, i_inputs.d_finalRegressor);
  }

  print(i_inputs);

  Math::Type lossSum = 0;
  Math::Type lossSumPrev = 0;
  std::ofstream predictionsFile;
  if (i_inputs.d_testonly)
  {
    predictionsFile.open(i_inputs.d_predictions.c_str());
  }

  std::cout << '\n' <<
      "Iteration num   | average loss  |  from last    |  lable        | yPredicted    | current t\n" <<
      "------------------------------------------------------------------------------------------------\n";

  PassWithTextFile(i_inputs, predictionsFile, hashes, count, lossSum, lossSumPrev, prevCount, sqrHashes);

  for (size_t pass = 0; pass + 1 < i_inputs.d_passes; ++pass)
  {
    PassWithCacheFile(i_inputs, predictionsFile, hashes, count, lossSum, lossSumPrev, prevCount, sqrHashes);
  }
  if (i_inputs.d_testonly)
  {
    predictionsFile.close();
  }
  if (!i_inputs.d_finalRegressor.empty() && !i_inputs.d_testonly)
  {
    WriteModel (i_inputs, hashes, i_inputs.d_finalRegressor);
  }

  print("");
  print("Run finished");
}
} // namespace Learning
