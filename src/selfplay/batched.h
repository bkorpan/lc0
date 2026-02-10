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

#include <functional>
#include <vector>

#include "chess/pgn.h"
#include "neural/backend.h"
#include "search/classic/node.h"
#include "search/classic/params.h"
#include "selfplay/game.h"
#include "syzygy/syzygy.h"
#include "trainingdata/trainingdata.h"

namespace lczero {

// Called to get the next opening. Returns true + populates opening/game_id
// if there's a game to play, false if no more games.
using NextOpeningCallback = std::function<bool(Opening*, int* game_id)>;

// Data passed to the game-finished callback.
struct FinishedGameData {
  int game_id;
  GameResult result;
  const V6TrainingDataArray& training_data;
  std::vector<Move> moves;
  Opening opening;
  int move_count;
  uint64_t nodes_total;
  bool adjudicated;
};

// Called when a game finishes.
using GameFinishedCallback = std::function<void(const FinishedGameData&)>;

class BatchedSelfPlay {
 public:
  BatchedSelfPlay(PlayerOptions player, int visits_per_move, int num_slots,
                  NextOpeningCallback next_opening,
                  GameFinishedCallback game_finished,
                  SyzygyTablebase* syzygy_tb);

  void Play();
  void Abort();

 private:
  struct GameState {
    std::shared_ptr<classic::NodeTree> tree;
    V6TrainingDataArray training_data;
    GameResult result = GameResult::UNDECIDED;
    int visits_this_move = 0;
    bool root_evaluated = false;
    int move_count = 0;
    uint64_t nodes_total = 0;
    Opening opening;
    int game_id = -1;
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

  void InitializeGame(GameState& game, const Opening& opening);
  std::vector<Move> GetMovesForGame(const GameState& game) const;
  void FinishAndRecycleGame(GameState& game);

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
  NextOpeningCallback next_opening_;
  GameFinishedCallback game_finished_;
};

}  // namespace lczero
