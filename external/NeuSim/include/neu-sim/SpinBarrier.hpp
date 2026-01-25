#pragma once

#include <atomic>
#include <cstddef>

// 为x86/x64架构提供CPU暂停指令的内在函数，对于自旋循环至关重要
#if defined(__i386__) || defined(__x86_64__)
#include <immintrin.h>
#define SPIN_HINT() _mm_pause()
#else
// 其他架构（如ARM）的回退方案
#define SPIN_HINT() __asm__ __volatile__("yield")
#endif

class SpinBarrier {
public:
  explicit SpinBarrier(size_t count)
    : m_initial_count(count), m_count(count), m_generation(0) {}

  // 删除拷贝和移动构造，确保其唯一性
  SpinBarrier(const SpinBarrier &) = delete;
  SpinBarrier &operator=(const SpinBarrier &) = delete;

  void arrive_and_wait() {
    // 读取当前的“代”，以便后续检查它是否变化
    const size_t current_gen = m_generation.load(std::memory_order_relaxed);

    // 原子地将计数器减1，并检查减之前的值
    if (m_count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      // ===================================
      // 这是最后一个到达的线程 (The Last Thread)
      // ===================================
      // 重置计数器，为下一轮屏障做准备
      m_count.store(m_initial_count, std::memory_order_relaxed);

      // 递增“代”，这会作为信号释放所有其他正在自旋等待的线程
      // `release`语义确保此操作前所有内存写入对其他线程可见
      m_generation.fetch_add(1, std::memory_order_release);
    } else {
      // ===================================
      // 其他等待的线程 (Waiting Threads)
      // ===================================
      // 进入自旋循环，忙等“代”发生变化
      // `acquire`语义确保一旦看到`generation`的变化，就能看到最后一个线程的所有写入
      while (m_generation.load(std::memory_order_acquire) == current_gen) {
        // SPIN_HINT() 是一个关键优化！
        // 它告诉CPU我们正在一个自旋等待循环中，
        // CPU可以借此机会节省功耗、避免流水线预测错误，并优先处理另一个超线程
        SPIN_HINT();
      }
    }
  }

private:
  const size_t m_initial_count;     // 屏障的参与者总数
  std::atomic<size_t> m_count;      // 当前轮次还需等待的线程数
  std::atomic<size_t> m_generation; // 屏障的“代”，用于释放和重用
};
