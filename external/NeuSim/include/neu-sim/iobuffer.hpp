#pragma once

#include <deque>
#include <memory>

template <typename T>
class IOBuffer {
private:
  std::deque<std::unique_ptr<T>> buf;
  std::unique_ptr<T> tmp { nullptr };
  uint32_t max_size { 1 };

public:
  IOBuffer() : max_size(1) {}

  IOBuffer(uint32_t size) : max_size(size) {}

  auto write_ready() -> bool { return tmp == nullptr; }

  auto write(std::unique_ptr<T> item) -> void { tmp = std::move(item); }

  auto flush() -> void {
    if (tmp != nullptr && buf.size() < max_size) {
      buf.push_back(std::move(tmp));
    }
  }

  auto read() -> T * {
    if (buf.empty())
      return nullptr;
    return buf.front().get();
  }

  auto take() -> std::unique_ptr<T> {
    if (buf.empty())
      return nullptr;
    auto item = std::move(buf.front());
    buf.pop_front();
    return item;
  }

  auto empty() -> bool { return buf.empty(); }

  auto is_clear() -> bool { return tmp == nullptr && buf.empty(); }
};
