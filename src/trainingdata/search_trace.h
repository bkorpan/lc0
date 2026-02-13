/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2024 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.

  Additional permission under GNU GPL version 3 section 7

  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/

#pragma once

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

namespace lczero {

// Writes MCTS search traces to a binary file.
//
// File format:
//   File Header (16 bytes):
//     magic:     uint32 = 0x5443484D ("MCHT")
//     version:   uint32 = 1
//     num_games: uint32
//     reserved:  uint32 = 0
//
//   Per Game:
//     Game Header (4 bytes):
//       num_plies: uint16
//       result:    int8 (1=white, -1=black, 0=draw)
//       reserved:  uint8
//
//     Game Moves (2 * num_plies bytes):
//       played_move: uint16[num_plies]  (NN index per ply)
//
//     Per Ply (num_plies entries):
//       Ply Header (2 bytes):
//         num_traces: uint16
//       Per Trace:
//         trace_len: uint8
//         moves:     uint16[trace_len]  (NN indices root->leaf)
class SearchTraceWriter {
 public:
  explicit SearchTraceWriter(const std::string& filename)
      : file_(filename, std::ios::binary) {
    // Write placeholder header; num_games filled in by Finalize().
    uint32_t header[4] = {kMagic, kVersion, 0, 0};
    file_.write(reinterpret_cast<const char*>(header), sizeof(header));
  }

  // Write a complete game's trace data to the file.
  void WriteGame(
      int8_t result, const std::vector<uint16_t>& played_moves,
      const std::vector<std::vector<std::vector<uint16_t>>>& ply_traces) {
    uint16_t num_plies = static_cast<uint16_t>(played_moves.size());
    uint8_t reserved = 0;

    // Game header.
    file_.write(reinterpret_cast<const char*>(&num_plies), 2);
    file_.write(reinterpret_cast<const char*>(&result), 1);
    file_.write(reinterpret_cast<const char*>(&reserved), 1);

    // Game moves.
    file_.write(reinterpret_cast<const char*>(played_moves.data()),
                num_plies * 2);

    // Per-ply traces.
    for (const auto& traces : ply_traces) {
      uint16_t num_traces = static_cast<uint16_t>(traces.size());
      file_.write(reinterpret_cast<const char*>(&num_traces), 2);
      for (const auto& trace : traces) {
        uint8_t trace_len = static_cast<uint8_t>(trace.size());
        file_.write(reinterpret_cast<const char*>(&trace_len), 1);
        file_.write(reinterpret_cast<const char*>(trace.data()),
                    trace_len * 2);
      }
    }

    num_games_++;
  }

  void Finalize() {
    // Seek back to write num_games in the header.
    file_.seekp(8);
    file_.write(reinterpret_cast<const char*>(&num_games_), 4);
    file_.close();
  }

 private:
  static constexpr uint32_t kMagic = 0x5443484D;  // "MCHT"
  static constexpr uint32_t kVersion = 1;

  std::ofstream file_;
  uint32_t num_games_ = 0;
};

}  // namespace lczero
