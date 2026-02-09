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

#include <vector>

#include "chess/pgn.h"
#include "neural/backend.h"
#include "search/classic/node.h"
#include "search/classic/params.h"
#include "selfplay/game.h"
#include "syzygy/syzygy.h"
#include "trainingdata/trainingdata.h"

namespace lczero {

class BatchedSelfPlay {
 public:
  BatchedSelfPlay(PlayerOptions player, int visits_per_move,
                  const std::vector<Opening>& openings,
                  SyzygyTablebase* syzygy_tb);

  void Play();
  void Abort();

  int NumGames() const { return static_cast<int>(games_.size()); }
  GameResult GetGameResult(int idx) const { return games_[idx].result; }
  std::vector<Move> GetMoves(int idx) const;
  void WriteTrainingData(int idx, TrainingDataWriter* writer) const;
  int GetMoveCount(int idx) const { return games_[idx].move_count; }
  uint64_t GetNodesTotal(int idx) const { return games_[idx].nodes_total; }

 private:
  struct GameState {
    std::shared_ptr<classic::NodeTree> tree;
    V6TrainingDataArray training_data;
    GameResult result = GameResult::UNDECIDED;
    int visits_this_move = 0;
    bool root_evaluated = false;
    int move_count = 0;
    uint64_t nodes_total = 0;
  };

  struct LeafToEval {
    int game_idx;
    classic::Node* leaf;
    std::vector<classic::Node*> path;
    PositionHistory history;
    std::vector<Move> legal_moves;
    EvalResult eval;
    bool is_root;
  };

  classic::Node* PuctWalk(GameState& game, std::vector<classic::Node*>& path,
                          PositionHistory& history);
  void ProcessNNResult(LeafToEval& leaf);
  void Backpropagate(const std::vector<classic::Node*>& path, float v, float d,
                     float m);
  void MakeGameMove(GameState& game);
  void CheckGameResult(GameState& game);

  PlayerOptions player_;
  classic::SearchParams params_;
  int visits_per_move_;
  SyzygyTablebase* syzygy_tb_;
  bool abort_ = false;
  bool has_mlh_ = false;
  std::vector<GameState> games_;
};

}  // namespace lczero
