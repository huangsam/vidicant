// File: filter.hpp
// Declarative filter expression parser and evaluator for Vidicant CLI.

#ifndef VIDICANT_CLI_FILTER_HPP
#define VIDICANT_CLI_FILTER_HPP

#include <algorithm>
#include <cctype>
#include <cmath>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace vidicant::cli::filter {

enum class TokenType {
  Identifier,
  Number,
  String,
  Boolean,
  OpGreater,
  OpGreaterEqual,
  OpLess,
  OpLessEqual,
  OpEqual,
  OpNotEqual,
  OpAnd,
  OpOr,
  OpNot,
  LParen,
  RParen,
  End
};

struct Token {
  TokenType type;
  std::string value;
  double numValue{0.0};
  bool boolValue{false};
};

inline std::vector<Token> tokenize(const std::string &expr) {
  std::vector<Token> tokens;
  size_t i = 0;
  while (i < expr.size()) {
    char c = expr[i];
    if (std::isspace(static_cast<unsigned char>(c))) {
      i++;
      continue;
    }
    if (c == '(') {
      tokens.push_back({TokenType::LParen, "("});
      i++;
    } else if (c == ')') {
      tokens.push_back({TokenType::RParen, ")"});
      i++;
    } else if (c == '!' && i + 1 < expr.size() && expr[i + 1] == '=') {
      tokens.push_back({TokenType::OpNotEqual, "!="});
      i += 2;
    } else if (c == '!') {
      tokens.push_back({TokenType::OpNot, "!"});
      i++;
    } else if (c == '=' && i + 1 < expr.size() && expr[i + 1] == '=') {
      tokens.push_back({TokenType::OpEqual, "=="});
      i += 2;
    } else if (c == '>') {
      if (i + 1 < expr.size() && expr[i + 1] == '=') {
        tokens.push_back({TokenType::OpGreaterEqual, ">="});
        i += 2;
      } else {
        tokens.push_back({TokenType::OpGreater, ">"});
        i++;
      }
    } else if (c == '<') {
      if (i + 1 < expr.size() && expr[i + 1] == '=') {
        tokens.push_back({TokenType::OpLessEqual, "<="});
        i += 2;
      } else {
        tokens.push_back({TokenType::OpLess, "<"});
        i++;
      }
    } else if (c == '&' && i + 1 < expr.size() && expr[i + 1] == '&') {
      tokens.push_back({TokenType::OpAnd, "&&"});
      i += 2;
    } else if (c == '|' && i + 1 < expr.size() && expr[i + 1] == '|') {
      tokens.push_back({TokenType::OpOr, "||"});
      i += 2;
    } else if (c == '"' || c == '\'') {
      char quote = c;
      i++;
      size_t start = i;
      while (i < expr.size() && expr[i] != quote) {
        i++;
      }
      std::string s = expr.substr(start, i - start);
      if (i < expr.size())
        i++;
      tokens.push_back({TokenType::String, s});
    } else if (std::isdigit(static_cast<unsigned char>(c)) ||
               (c == '-' && i + 1 < expr.size() &&
                std::isdigit(static_cast<unsigned char>(expr[i + 1])))) {
      size_t start = i;
      if (c == '-')
        i++;
      while (i < expr.size() &&
             (std::isdigit(static_cast<unsigned char>(expr[i])) ||
              expr[i] == '.')) {
        i++;
      }
      std::string numStr = expr.substr(start, i - start);
      double val = std::stod(numStr);
      Token t{TokenType::Number, numStr};
      t.numValue = val;
      tokens.push_back(t);
    } else if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
      size_t start = i;
      while (i < expr.size() &&
             (std::isalnum(static_cast<unsigned char>(expr[i])) ||
              expr[i] == '_')) {
        i++;
      }
      std::string ident = expr.substr(start, i - start);
      std::string lower = ident;
      std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
      if (lower == "and") {
        tokens.push_back({TokenType::OpAnd, "and"});
      } else if (lower == "or") {
        tokens.push_back({TokenType::OpOr, "or"});
      } else if (lower == "not") {
        tokens.push_back({TokenType::OpNot, "not"});
      } else if (lower == "true") {
        Token t{TokenType::Boolean, "true"};
        t.boolValue = true;
        tokens.push_back(t);
      } else if (lower == "false") {
        Token t{TokenType::Boolean, "false"};
        t.boolValue = false;
        tokens.push_back(t);
      } else {
        tokens.push_back({TokenType::Identifier, ident});
      }
    } else {
      i++;
    }
  }
  tokens.push_back({TokenType::End, ""});
  return tokens;
}

class Evaluator {
public:
  explicit Evaluator(std::vector<Token> tokens, const nlohmann::json &record)
      : tokens_(std::move(tokens)), pos_(0), record_(record) {}

  bool evaluate() {
    if (tokens_.empty() || tokens_[0].type == TokenType::End)
      return true;
    return parseOr();
  }

private:
  std::vector<Token> tokens_;
  size_t pos_{0};
  const nlohmann::json &record_;

  const Token &peek() const {
    if (pos_ < tokens_.size())
      return tokens_[pos_];
    return tokens_.back();
  }

  Token get() {
    if (pos_ < tokens_.size())
      return tokens_[pos_++];
    return tokens_.back();
  }

  bool parseOr() {
    bool left = parseAnd();
    while (peek().type == TokenType::OpOr) {
      get();
      bool right = parseAnd();
      left = left || right;
    }
    return left;
  }

  bool parseAnd() {
    bool left = parseUnary();
    while (peek().type == TokenType::OpAnd) {
      get();
      bool right = parseUnary();
      left = left && right;
    }
    return left;
  }

  bool parseUnary() {
    if (peek().type == TokenType::OpNot) {
      get();
      return !parseUnary();
    }
    return parsePrimary();
  }

  bool parsePrimary() {
    if (peek().type == TokenType::LParen) {
      get();
      bool val = parseOr();
      if (peek().type == TokenType::RParen) {
        get();
      }
      return val;
    }
    return parseComparison();
  }

  bool parseComparison() {
    Token lhs = get();
    if (lhs.type != TokenType::Identifier) {
      if (lhs.type == TokenType::Boolean)
        return lhs.boolValue;
      return false;
    }

    TokenType opType = peek().type;
    if (opType == TokenType::OpGreater || opType == TokenType::OpGreaterEqual ||
        opType == TokenType::OpLess || opType == TokenType::OpLessEqual ||
        opType == TokenType::OpEqual || opType == TokenType::OpNotEqual) {
      get();
      Token rhs = get();
      return evaluateBinary(lhs.value, opType, rhs);
    }

    if (record_.contains(lhs.value) && record_[lhs.value].is_boolean()) {
      return record_[lhs.value].get<bool>();
    }
    return false;
  }

  bool evaluateBinary(const std::string &field, TokenType op,
                      const Token &rhs) {
    if (!record_.contains(field))
      return false;
    const auto &val = record_[field];
    if (val.is_null())
      return false;

    if (val.is_number()) {
      double l = val.get<double>();
      double r = 0.0;
      if (rhs.type == TokenType::Number) {
        r = rhs.numValue;
      } else if (rhs.type == TokenType::Identifier &&
                 record_.contains(rhs.value) &&
                 record_[rhs.value].is_number()) {
        r = record_[rhs.value].get<double>();
      } else {
        return false;
      }

      switch (op) {
      case TokenType::OpGreater:
        return l > r;
      case TokenType::OpGreaterEqual:
        return l >= r;
      case TokenType::OpLess:
        return l < r;
      case TokenType::OpLessEqual:
        return l <= r;
      case TokenType::OpEqual:
        return std::abs(l - r) < 1e-6;
      case TokenType::OpNotEqual:
        return std::abs(l - r) >= 1e-6;
      default:
        return false;
      }
    } else if (val.is_boolean()) {
      bool l = val.get<bool>();
      bool r = false;
      if (rhs.type == TokenType::Boolean) {
        r = rhs.boolValue;
      } else {
        return false;
      }
      if (op == TokenType::OpEqual)
        return l == r;
      if (op == TokenType::OpNotEqual)
        return l != r;
      return false;
    } else if (val.is_string()) {
      std::string l = val.get<std::string>();
      std::string r = rhs.value;
      if (op == TokenType::OpEqual)
        return l == r;
      if (op == TokenType::OpNotEqual)
        return l != r;
      return false;
    }
    return false;
  }
};

inline bool evaluateFilter(const nlohmann::json &record,
                           const std::string &expr) {
  if (expr.empty())
    return true;
  auto tokens = tokenize(expr);
  Evaluator eval(tokens, record);
  return eval.evaluate();
}

} // namespace vidicant::cli::filter

#endif // VIDICANT_CLI_FILTER_HPP
