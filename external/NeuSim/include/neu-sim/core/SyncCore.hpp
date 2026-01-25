#pragma once

#include "neu-sim/core/BaseCore.hpp"
#include "neu-sim/noc/packet.hpp"

#include <cassert>
#include <deque>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

/**
 * SyncCore: A synchronous core implementation with barrier synchronization.
 *
 * This core implements a tree-based barrier synchronization protocol:
 * - Each core has parent(s) and children in the synchronization tree
 * - Sync packets flow up from children to parents
 * - Commit packets flow down from parents to children
 *
 * State machine:
 *   Idle -> Neuron -> End -> SyncWait -> Commit -> (next timestep or Idle)
 */
class SyncCore : public BaseCore {
public:
  class SyncPacket : public Packet {
  public:
    enum PacketType : uint32_t {
      Spike = 0,  // Regular spike packet
      Sync = 1,   // Synchronization packet (child -> parent)
      Commit = 2, // Commit packet (parent -> child)
    };

    PacketType ptype { PacketType::Spike };
    uint32_t ts { 0 };        // Timestamp
    uint32_t neuron_id { 0 }; // Neuron ID for spike packets
    uint32_t src_lid { 0 };   // Source logical ID
    uint32_t dst_lid { 0 };   // Destination logical ID
    bool is_anti { false };   // Anti-spike flag

    SyncPacket(uint32_t src, uint32_t dst, uint32_t len)
      : Packet(src, dst, len) {}

    static std::unique_ptr<SyncPacket> new_packet(
      uint32_t src, uint32_t dst, uint32_t len, PacketType type, uint32_t ts
    ) {
      auto pkt = std::make_unique<SyncPacket>(src, dst, len);
      pkt->ptype = type;
      pkt->ts = ts;
      return pkt;
    }
  };

  /**
   * Buffer for queuing neurons to be processed
   */
  struct NeuronBuffer {
    uint32_t pc { 0 }; // Program counter (current neuron index)
    uint32_t buf_size; // Maximum buffer size
    std::deque<uint32_t> buf;

    NeuronBuffer(uint32_t buf_size) : buf_size(buf_size) {}

    void push_back(uint32_t n) { buf.emplace_back(n); }

    uint32_t pop_front() {
      auto res = buf.front();
      buf.pop_front();
      return res;
    }

    bool is_empty() const { return buf.empty(); }

    void reset() {
      pc = 0;
      buf.clear();
    }
  };

  /**
   * Core state machine
   */
  struct CoreState {
    enum State {
      Idle,     // Core is idle, simulation finished
      Neuron,   // Processing neurons
      SyncWait, // Waiting for commit from parents
      Commit    // Sending commit to children
    } cur { Idle };

    struct {
      std::optional<uint32_t> fire;  // Current firing neuron
      uint32_t fire_cnt { 0 };       // Number of spikes sent for current neuron
      std::optional<uint32_t> stall; // Stalled firing neuron (waiting)
    } neuron;

    uint32_t recv_sync_cnt { 0 };   // Received sync packets count
    uint32_t recv_commit_cnt { 0 }; // Received commit packets count
    uint32_t send_sync_cnt { 0 };   // Sent sync packets count
    uint32_t send_commit_cnt { 0 }; // Sent commit packets count

    std::string_view fmt() const {
      switch (cur) {
      case Idle:
        return "Idle";
      case Neuron:
        return "Neuron";
      case SyncWait:
        return "SyncWait";
      case Commit:
        return "Commit";
      default:
        return "Unknown";
      }
    }

    void reset() {
      cur = Idle;
      neuron = { std::nullopt, 0, std::nullopt };
      recv_sync_cnt = 0;
      recv_commit_cnt = 0;
      send_sync_cnt = 0;
      send_commit_cnt = 0;
    }
  };

  // Synchronization tree structure
  std::vector<uint32_t> parents_id;  // Parent core IDs
  std::vector<uint32_t> children_id; // Children core IDs

  // Core state
  CoreState state;
  NeuronBuffer neuron_buf;

  // On-chip memory
  std::vector<uint32_t> neurons;

  // std::vector<uint32_t> weightsum;
  struct Axon {
    std::vector<uint32_t> dst_cores;
  };

  std::vector<Axon> axons;

  // struct Dendrite {
  //   // Placeholder for dendrite parameters
  // };
  // std::vector<Dendrite> dendrites;

  // Timestep management
  uint32_t m_ts { 0 };
  uint32_t m_max_ts { 0 };

  // Parameters
  bool sparse { false };
  // uint32_t logical_id { 0 };

  // Only for simulation
  // data
  std::vector<std::span<const uint8_t>> spike_table;

  // Statistics
  struct Stats {
    uint32_t total_recv_spikes { 0 };
    uint32_t total_sent_spikes { 0 };
    uint32_t update_cnt { 0 };
    uint32_t firing_cnt { 0 };
    uint32_t packet_cnt { 0 };
    std::vector<uint32_t> recv_spike_cnt;
    std::vector<uint32_t> start_cycle;
    std::vector<uint32_t> end_cycle;
    std::vector<uint32_t> neuron_cycle_cnt;
    std::vector<uint32_t> neuron_redundant_cnt;
    std::vector<uint32_t> neuron_has_input_cnt;
    std::vector<uint32_t> neuron_has_state_cnt;
  } stats;

public:
  SyncCore(ClockDomain &clk, uint32_t id)
    : BaseCore(clk, id), neuron_buf(global_params.buf_size) {}

  /**
   * Initialize the core for simulation
   */
  auto init(uint32_t max_ts) -> void {
    m_max_ts = max_ts;
    m_ts = 0;
    state.reset();
    neuron_buf.reset();

    // Initialize statistics vectors
    stats.recv_spike_cnt.resize(max_ts, 0);
    stats.start_cycle.resize(max_ts, 0);
    stats.end_cycle.resize(max_ts, 0);
    stats.neuron_cycle_cnt.resize(max_ts, 0);
    stats.neuron_redundant_cnt.resize(max_ts, 0);
    stats.neuron_has_input_cnt.resize(max_ts, 0);
    stats.neuron_has_state_cnt.resize(max_ts, 0);
  }

  /**
   * Set synchronization tree structure
   */
  auto
  set_sync_tree(std::vector<uint32_t> parents, std::vector<uint32_t> children)
    -> void {
    parents_id = std::move(parents);
    children_id = std::move(children);
  }

  auto
  add_neuron(std::span<const uint8_t> axon, std::span<const uint8_t> spikes)
    -> void {
    neurons.push_back(0);
    auto dst_cores = std::vector<uint32_t>();
    for (uint32_t i = 0; i < axon.size(); ++i) {
      if (axon[i]) {
        dst_cores.push_back(i);
      }
    }
    axons.push_back({ .dst_cores = dst_cores });
    spike_table.push_back(spikes);
  }

  auto launch() -> void { next_ts(); }

  auto is_stop() -> bool override {
    return state.cur == CoreState::Idle && BaseCore::is_stop();
  }

  auto cur_ts() const -> uint32_t { return m_ts; }

  auto max_ts() const -> uint32_t { return m_max_ts; }

  auto cur_cycle() const -> uint32_t { return clk.cycle_count; }

  auto get_state() const -> std::string_view { return state.fmt(); }

protected:
  /**
   * Main cycle function - implements the state machine
   */
  auto core_cycle() -> void override {
    // Process incoming packet
    process_packet();

    // State machine
    switch (state.cur) {
    case CoreState::Neuron:
      do_neuron();
      break;
    case CoreState::SyncWait:
      do_syncwait();
      break;
    case CoreState::Commit:
      do_commit();
      break;
    case CoreState::Idle:
    default:
      break;
    }
  }

  /**
   * Process received packet
   */
  auto process_packet() -> void {
    auto *base_packet = get_packet();
    if (base_packet == nullptr)
      return;

    auto *packet = dynamic_cast<SyncPacket *>(base_packet);

    switch (packet->ptype) {
    case SyncPacket::Spike:
      process_spike(packet);
      break;
    case SyncPacket::Sync:
      state.recv_sync_cnt += 1;
      break;
    case SyncPacket::Commit:
      state.recv_commit_cnt += 1;
      break;
    }
    consume_packet();
  }

  /**
   * Handle received spike packet
   * Override this to implement spike processing logic
   */
  auto process_spike(SyncPacket *packet) -> void {
    auto ts = packet->ts;
    stats.recv_spike_cnt[ts - 1] += 1;
    stats.total_recv_spikes += 1;
    // Subclass should implement actual spike processing
  }

  /**
   * Check if neuron is firing at current timestep
   * Override this to implement firing logic
   */
  auto is_firing(uint32_t ts, uint32_t neuron_id) -> bool {
    // if (spike_table[neuron_id][ts - 1]) {
    //   printf("ts %d: Core %d Neuron %d fires\n", ts, id, neuron_id);
    // }
    return spike_table[neuron_id][ts - 1]; // Default: no firing
  }

  /**
   * Check if neuron can be skipped (redundant)
   * Override this to implement sparse optimization
   */
  auto is_redundant(uint32_t ts, uint32_t neuron_id) -> bool {
    return false; // Default: not redundant
  }

  /**
   * Update neuron state
   * Override this to implement neuron update logic
   */
  auto update_neuron(uint32_t ts, uint32_t neuron_id) -> void {
    // Subclass should implement actual neuron update
    stats.update_cnt += 1;
  }

  /**
   * Get number of destination cores for a neuron's spikes
   * Override this to implement connectivity
   */
  auto get_spike_dst_count(uint32_t neuron_id) -> uint32_t {
    // printf("Core %d Neuron %d has %ld destinations\n", id, neuron_id,
    // axons[neuron_id].dst_cores.size());
    return axons[neuron_id].dst_cores.size(); // Default: no destinations
  }

  /**
   * Get destination core ID for a neuron's spike
   * Override this to implement connectivity
   */
  auto get_spike_dst(uint32_t neuron_id, uint32_t index) -> uint32_t {
    return axons[neuron_id].dst_cores[index];
  }

private:
  /**
   * Advance to next timestep
   */
  auto next_ts() -> void {
    if (m_ts >= m_max_ts) {
      state.cur = CoreState::Idle;
      return;
    }

    m_ts += 1;
    state.cur = CoreState::Neuron;
    state.neuron = { std::nullopt, 0, std::nullopt };

    // stats.firing_cnt = 0;
    stats.packet_cnt = 0;
    stats.start_cycle[m_ts - 1] = cur_cycle();
  }

  /**
   * Finish current timestep
   */
  auto finish_ts() -> void {
    if (m_ts > 0 && m_ts <= m_max_ts) {
      stats.end_cycle[m_ts - 1] = cur_cycle();
    }
  }

  /**
   * Send spike to destination core
   */
  auto send_spike(uint32_t ts, uint32_t dst, uint32_t neuron_id) -> bool {
    // Self-loop: direct processing
    if (dst == id) {
      stats.total_sent_spikes += 1;
      auto packet = gen_spike(ts, dst, neuron_id);
      process_spike(packet.get());
      return true;
    }

    if (send_packet_ready()) {
      stats.total_sent_spikes += 1;
      auto packet = gen_spike(ts, dst, neuron_id);
      send_packet(std::move(packet));
      stats.packet_cnt += 1;
      return true;
    }
    return false;
  }

  /**
   * Generate spike packet
   */
  auto gen_spike(uint32_t ts, uint32_t dst, uint32_t neuron_id)
    -> std::unique_ptr<SyncPacket> {
    auto packet = SyncPacket::new_packet(
      id, dst, global_params.packet_size, SyncPacket::Spike, ts
    );
    packet->neuron_id = neuron_id;
    // packet->src_lid = logical_id;
    // packet->dst_lid = dst;
    return packet;
  }

  /**
   * Send sync packet to parent
   */
  auto send_sync(uint32_t parent_idx) -> bool {
    if (send_packet_ready()) {
      send_packet(gen_sync(parent_idx));
      return true;
    }
    return false;
  }

  /**
   * Generate sync packet
   */
  auto gen_sync(uint32_t parent_idx) -> std::unique_ptr<SyncPacket> {
    auto dst = parents_id[parent_idx];
    auto packet = SyncPacket::new_packet(
      id, dst, global_params.packet_size, SyncPacket::Sync, m_ts
    );
    return packet;
  }

  /**
   * Send commit packet to child
   */
  auto send_commit(uint32_t child_idx) -> bool {
    if (send_packet_ready()) {
      send_packet(gen_commit(child_idx));
      return true;
    }
    return false;
  }

  /**
   * Generate commit packet
   */
  auto gen_commit(uint32_t child_idx) -> std::unique_ptr<SyncPacket> {
    auto dst = children_id[child_idx];
    auto packet = SyncPacket::new_packet(
      id, dst, global_params.packet_size, SyncPacket::Commit, m_ts
    );
    return packet;
  }

  /**
   * State: Neuron processing
   */
  auto do_neuron() -> void {
    if (neurons.empty()) {
      // Skip to end if no neurons
      state.cur = CoreState::SyncWait;
      finish_ts();
      do_syncwait();
      return;
    }

    // Handle ongoing spike transmission
    if (state.neuron.fire.has_value()) {
      auto neuron = state.neuron.fire.value();
      auto n = state.neuron.fire_cnt;
      auto dst_count = get_spike_dst_count(neuron);

      if (n < dst_count) {
        auto dst = get_spike_dst(neuron, n);
        if (send_spike(m_ts, dst, neuron)) {
          n += 1;
        }
      }

      if (n < dst_count) {
        state.neuron.fire_cnt = n;
      } else {
        state.neuron.fire.reset();
        state.neuron.fire_cnt = 0;
      }
    } else {
      assert(state.neuron.fire_cnt == 0);
    }

    // Handle stalled neuron
    if (state.neuron.stall.has_value()) {
      if (!state.neuron.fire.has_value()) {
        state.neuron.fire = state.neuron.stall;
        state.neuron.stall.reset();
      }
    }
    if (!state.neuron.stall.has_value()) {
      // Neuron update
      if (!neuron_buf.is_empty()) {
        auto neuron = neuron_buf.pop_front();
        update_neuron(m_ts, neuron);

        if (m_ts > 0 && m_ts <= m_max_ts) {
          stats.neuron_cycle_cnt[m_ts - 1] += 1;
          if (is_redundant(m_ts, neuron)) {
            stats.neuron_redundant_cnt[m_ts - 1] += 1;
          }
        }

        if (is_firing(m_ts, neuron)) {
          stats.firing_cnt += 1;
          if (!state.neuron.fire.has_value())
            state.neuron.fire = neuron;
          else
            state.neuron.stall = neuron;
        }
      } else if (m_ts > 0 && m_ts <= m_max_ts) {
        stats.neuron_cycle_cnt[m_ts - 1] += 1;
        stats.neuron_redundant_cnt[m_ts - 1] += 1;
      }
    }

    // Fetch more neurons
    fetch_neuron();

    // Check if neuron phase is complete
    if (neuron_buf.is_empty() && neuron_buf.pc == neurons.size() &&
        !state.neuron.fire.has_value() && !state.neuron.stall.has_value()) {
      neuron_buf.pc = 0;
      state.cur = CoreState::SyncWait;
      finish_ts();
    }
  }

  /**
   * Fetch neurons into buffer
   */
  auto fetch_neuron() -> void {
    auto fetch_num = neurons.size() - neuron_buf.pc;
    if (fetch_num > neuron_buf.buf_size)
      fetch_num = neuron_buf.buf_size;

    for (uint32_t i = 0; i < fetch_num; ++i) {
      if (neuron_buf.buf.size() >= neuron_buf.buf_size)
        break;

      auto neuron_id = neuron_buf.pc;
      if (sparse && is_redundant(m_ts, neuron_id)) {
        neuron_buf.pc += 1;
        continue;
      }
      neuron_buf.push_back(neuron_id);
      neuron_buf.pc += 1;
    }
  }

  /**
   * State: SyncWait - send sync to parents
   */
  auto do_syncwait() -> void {
    // Wait for all children to sync
    if (state.recv_sync_cnt != children_id.size())
      return;

    // If no parents, go directly to SyncWait
    if (parents_id.empty()) {
      state.recv_sync_cnt = 0;
      state.cur = CoreState::Commit;
      return;
    }

    // Send sync to parents
    assert(state.send_sync_cnt < parents_id.size());
    if (send_sync(state.send_sync_cnt)) {
      if (state.send_sync_cnt + 1 < parents_id.size()) {
        state.send_sync_cnt += 1;
      } else {
        state.send_sync_cnt = 0;
        state.recv_sync_cnt = 0;
        state.cur = CoreState::Commit;
      }
    }
  }

  /**
   * State: Commit - send commit to children
   */
  auto do_commit() -> void {
    if (state.recv_commit_cnt != parents_id.size()) {
      return;
    }

    // If no children, go to next timestep
    if (children_id.empty()) {
      state.recv_commit_cnt = 0;
      next_ts();
      return;
    }

    // Send commit to children
    assert(state.send_commit_cnt < children_id.size());
    if (send_commit(state.send_commit_cnt)) {
      if (state.send_commit_cnt + 1 < children_id.size()) {
        state.send_commit_cnt += 1;
      } else {
        state.recv_commit_cnt = 0;
        state.send_commit_cnt = 0;
        next_ts();
      }
    }
  }
};
