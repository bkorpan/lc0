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

#include "selfplay/batched.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "utils/fastmath.h"
#include "utils/random.h"

namespace lczero {

namespace {

// Reimplemented from search.cc (anonymous namespace there).
inline float GetFpu(const classic::SearchParams& params,
                    const classic::Node* node, bool is_root_node,
                    float draw_score) {
  const auto value = params.GetFpuValue(is_root_node);
  return params.GetFpuAbsolute(is_root_node)
             ? value
             : -node->GetQ(-draw_score) -
                   value * std::sqrt(node->GetVisitedPolicy());
}

inline float ComputeCpuct(const classic::SearchParams& params, uint32_t N,
                          bool is_root_node) {
  const float init = params.GetCpuct(is_root_node);
  const float k = params.GetCpuctFactor(is_root_node);
  const float base = params.GetCpuctBase(is_root_node);
  return init + (k ? k * FastLog((N + base) / base) : 0.0f);
}

void ApplyDirichletNoise(classic::Node* node, float eps, double alpha) {
  float total = 0;
  std::vector<float> noise;
  for (int i = 0; i < node->GetNumEdges(); ++i) {
    float eta = Random::Get().GetGamma(alpha, 1.0);
    noise.emplace_back(eta);
    total += eta;
  }
  if (total < std::numeric_limits<float>::min()) return;
  int noise_idx = 0;
  for (const auto& child : node->Edges()) {
    auto* edge = child.edge();
    edge->SetP(edge->GetP() * (1 - eps) + eps * noise[noise_idx++] / total);
  }
}

// Copied from search.cc (anonymous namespace there).
class MEvaluator {
 public:
  MEvaluator()
      : enabled_{false},
        m_slope_{0.0f},
        m_cap_{0.0f},
        a_constant_{0.0f},
        a_linear_{0.0f},
        a_square_{0.0f},
        q_threshold_{0.0f},
        parent_m_{0.0f} {}

  MEvaluator(const classic::SearchParams& params,
             const classic::Node* parent = nullptr)
      : enabled_{true},
        m_slope_{params.GetMovesLeftSlope()},
        m_cap_{params.GetMovesLeftMaxEffect()},
        a_constant_{params.GetMovesLeftConstantFactor()},
        a_linear_{params.GetMovesLeftScaledFactor()},
        a_square_{params.GetMovesLeftQuadraticFactor()},
        q_threshold_{params.GetMovesLeftThreshold()},
        parent_m_{parent ? parent->GetM() : 0.0f},
        parent_within_threshold_{parent ? WithinThreshold(parent, q_threshold_)
                                        : false} {}

  void SetParent(const classic::Node* parent) {
    assert(parent);
    if (enabled_) {
      parent_m_ = parent->GetM();
      parent_within_threshold_ = WithinThreshold(parent, q_threshold_);
    }
  }

  float GetMUtility(classic::Node* child, float q) const {
    if (!enabled_ || !parent_within_threshold_) return 0.0f;
    const float child_m = child->GetM();
    float m = std::clamp(m_slope_ * (child_m - parent_m_), -m_cap_, m_cap_);
    m *= FastSign(-q);
    if (q_threshold_ > 0.0f && q_threshold_ < 1.0f) {
      q = std::max(0.0f, (std::abs(q) - q_threshold_)) / (1.0f - q_threshold_);
    }
    m *= a_constant_ + a_linear_ * std::abs(q) + a_square_ * q * q;
    return m;
  }

  float GetDefaultMUtility() const { return 0.0f; }

 private:
  static bool WithinThreshold(const classic::Node* parent, float q_threshold) {
    return std::abs(parent->GetQ(0.0f)) > q_threshold;
  }

  const bool enabled_;
  const float m_slope_;
  const float m_cap_;
  const float a_constant_;
  const float a_linear_;
  const float a_square_;
  const float q_threshold_;
  float parent_m_ = 0.0f;
  bool parent_within_threshold_ = false;
};

}  // namespace

BatchedSelfPlay::BatchedSelfPlay(PlayerOptions player, int visits_per_move,
                                 int num_slots,
                                 NextOpeningCallback next_opening,
                                 GameFinishedCallback game_finished,
                                 SyzygyTablebase* syzygy_tb)
    : player_(player),
      params_(*player.uci_options),
      visits_per_move_(visits_per_move),
      syzygy_tb_(syzygy_tb),
      next_opening_(std::move(next_opening)),
      game_finished_(std::move(game_finished)) {
  has_mlh_ = player_.backend->GetAttributes().has_mlh;
  games_.reserve(num_slots);
  for (int i = 0; i < num_slots; i++) {
    Opening opening;
    int game_id;
    if (!next_opening_(&opening, &game_id)) break;
    games_.push_back(GameState{
        .tree = std::make_shared<classic::NodeTree>(),
        .training_data = V6TrainingDataArray(
            params_.GetHistoryFill(), params_.GetHistoryFill(),
            pblczero::NetworkFormat::INPUT_CLASSICAL_112_PLANE),
        .opening = opening,
        .game_id = game_id,
    });
    auto& game = games_.back();
    game.tree->ResetToPosition(opening.start_fen, {});
    for (Move m : opening.moves) {
      if (game.tree->IsBlackToMove()) m.Flip();
      game.tree->MakeMove(m);
    }
  }
}

void BatchedSelfPlay::InitializeGame(GameState& game,
                                     const Opening& opening) {
  game.tree->ResetToPosition(opening.start_fen, {});
  for (Move m : opening.moves) {
    if (game.tree->IsBlackToMove()) m.Flip();
    game.tree->MakeMove(m);
  }
  game.training_data = V6TrainingDataArray(
      params_.GetHistoryFill(), params_.GetHistoryFill(),
      pblczero::NetworkFormat::INPUT_CLASSICAL_112_PLANE);
  game.result = GameResult::UNDECIDED;
  game.visits_this_move = 0;
  game.root_evaluated = false;
  game.move_count = 0;
  game.nodes_total = 0;
  game.opening = opening;
}

std::vector<Move> BatchedSelfPlay::GetMovesForGame(
    const GameState& game) const {
  std::vector<Move> moves;
  bool flip = !game.tree->IsBlackToMove();
  for (classic::Node* node = game.tree->GetCurrentHead();
       node != game.tree->GetGameBeginNode(); node = node->GetParent()) {
    moves.push_back(node->GetParent()->GetEdgeToNode(node)->GetMove(flip));
    flip = !flip;
  }
  std::reverse(moves.begin(), moves.end());
  return moves;
}

void BatchedSelfPlay::FinishAndRecycleGame(GameState& game) {
  bool adjudicated =
      (game.result == GameResult::DRAW &&
       game.tree->GetPositionHistory().Last().GetGamePly() >= 450);
  FinishedGameData data{
      .game_id = game.game_id,
      .result = game.result,
      .training_data = game.training_data,
      .moves = GetMovesForGame(game),
      .opening = game.opening,
      .move_count = game.move_count,
      .nodes_total = game.nodes_total,
      .adjudicated = adjudicated,
  };
  game_finished_(data);

  Opening opening;
  int game_id;
  if (next_opening_(&opening, &game_id)) {
    game.game_id = game_id;
    InitializeGame(game, opening);
  }
  // If next_opening_ returns false, game.result stays non-UNDECIDED and
  // the slot will be skipped in future iterations.
}

void BatchedSelfPlay::Play() {
  while (true) {
    if (abort_) break;

    bool any_active = false;
    auto comp = player_.backend->CreateComputation();
    std::vector<LeafToEval> pending_leaves;

    for (int gi = 0; gi < static_cast<int>(games_.size()); gi++) {
      auto& game = games_[gi];
      if (game.result != GameResult::UNDECIDED) continue;

      CheckGameResult(game);
      if (game.result != GameResult::UNDECIDED) {
        FinishAndRecycleGame(game);
        if (game.result != GameResult::UNDECIDED) continue;
      }

      // If enough visits, pick and play a move.
      if (game.visits_this_move >= visits_per_move_) {
        MakeGameMove(game);
        game.visits_this_move = 0;
        game.root_evaluated = false;
        CheckGameResult(game);
        if (game.result != GameResult::UNDECIDED) {
          FinishAndRecycleGame(game);
          if (game.result != GameResult::UNDECIDED) continue;
        }
      }

      any_active = true;

      // Do one MCTS iteration: walk tree to find leaf.
      std::vector<classic::Node*> path;
      PositionHistory history = game.tree->GetPositionHistory();
      classic::Node* leaf = PuctWalk(game, path, history);

      if (leaf->IsTerminal()) {
        Backpropagate(path, leaf->GetWL(), leaf->GetD(), leaf->GetM());
        game.visits_this_move++;
      } else {
        // Need NN eval: expand leaf, add to batch.
        auto& board = history.Last().GetBoard();
        auto legal_moves = board.GenerateLegalMoves();

        // If no legal moves, this is a terminal node (checkmate/stalemate).
        // Use WHITE_WON for checkmate (= "parent won") to match node value
        // convention used by search.cc, NOT ComputeGameResult() which gives
        // absolute color results.
        if (legal_moves.empty()) {
          if (board.IsUnderCheck()) {
            leaf->MakeTerminal(GameResult::WHITE_WON);
          } else {
            leaf->MakeTerminal(GameResult::DRAW);
          }
          Backpropagate(path, leaf->GetWL(), leaf->GetD(), leaf->GetM());
          game.visits_this_move++;
          continue;
        }

        // Draw-by-rule and TwoFold draw detection (matching ExtendNode in
        // search.cc). Only for non-root nodes.
        if (leaf != game.tree->GetCurrentHead()) {
          if (!board.HasMatingMaterial()) {
            leaf->MakeTerminal(GameResult::DRAW);
            Backpropagate(path, leaf->GetWL(), leaf->GetD(), leaf->GetM());
            game.visits_this_move++;
            continue;
          }
          if (history.Last().GetRule50Ply() >= 100) {
            leaf->MakeTerminal(GameResult::DRAW);
            Backpropagate(path, leaf->GetWL(), leaf->GetD(), leaf->GetM());
            game.visits_this_move++;
            continue;
          }
          const auto repetitions = history.Last().GetRepetitions();
          if (repetitions >= 2) {
            leaf->MakeTerminal(GameResult::DRAW);
            Backpropagate(path, leaf->GetWL(), leaf->GetD(), leaf->GetM());
            game.visits_this_move++;
            continue;
          } else if (repetitions == 1 && params_.GetTwoFoldDraws()) {
            const int depth = static_cast<int>(path.size()) - 1;
            const auto cycle_length =
                history.Last().GetPliesSincePrevRepetition();
            if (depth >= 4 && depth >= cycle_length) {
              leaf->MakeTerminal(GameResult::DRAW,
                                 static_cast<float>(cycle_length),
                                 classic::Node::Terminal::TwoFold);
              Backpropagate(path, leaf->GetWL(), leaf->GetD(), leaf->GetM());
              game.visits_this_move++;
              continue;
            }
          }
        }

        pending_leaves.push_back(LeafToEval{
            .game_idx = gi,
            .leaf = leaf,
            .path = std::move(path),
            .history = history,
            .legal_moves = std::move(legal_moves),
            .eval = {},
            .is_root = (leaf == game.tree->GetCurrentHead()),
        });
      }
    }

    if (!any_active) break;

    // Second pass: now the vector is stable, create edges and add to batch.
    // We must do this after the vector is fully built so that pointers into
    // EvalResult (via AsPtr()) are not invalidated by reallocation.
    for (auto& leaf_info : pending_leaves) {
      if (!leaf_info.leaf->HasChildren()) {
        leaf_info.leaf->CreateEdges(MoveList(leaf_info.legal_moves.begin(),
                                             leaf_info.legal_moves.end()));
      }
      leaf_info.eval.p.resize(leaf_info.legal_moves.size());
      auto result = comp->AddInput(
          EvalPosition{.pos = leaf_info.history.GetPositions(),
                       .legal_moves = leaf_info.legal_moves},
          leaf_info.eval.AsPtr());
      if (result == BackendComputation::FETCHED_IMMEDIATELY) {
        ProcessNNResult(leaf_info);
        Backpropagate(leaf_info.path, -leaf_info.eval.q, leaf_info.eval.d,
                      leaf_info.eval.m);
        games_[leaf_info.game_idx].visits_this_move++;
        leaf_info.game_idx = -1;  // Mark as already processed.
      }
    }

    // Only call ComputeBlocking if there are actually enqueued evals.
    bool has_enqueued = false;
    for (auto& leaf_info : pending_leaves) {
      if (leaf_info.game_idx >= 0) {
        has_enqueued = true;
        break;
      }
    }
    if (has_enqueued) {
      comp->ComputeBlocking();
      for (auto& leaf_info : pending_leaves) {
        if (leaf_info.game_idx < 0) continue;
        ProcessNNResult(leaf_info);
        Backpropagate(leaf_info.path, -leaf_info.eval.q, leaf_info.eval.d,
                      leaf_info.eval.m);
        games_[leaf_info.game_idx].visits_this_move++;
      }
    }
  }
}

void BatchedSelfPlay::Abort() { abort_ = true; }

classic::Node* BatchedSelfPlay::PuctWalk(
    GameState& game, std::vector<classic::Node*>& path,
    PositionHistory& history) {
  auto* node = game.tree->GetCurrentHead();
  path.push_back(node);

  const float draw_score = params_.GetDrawScore();
  auto m_evaluator = has_mlh_ ? MEvaluator(params_) : MEvaluator();

  while (node->GetN() > 0 && !node->IsTerminal()) {
    bool at_root = (node == game.tree->GetCurrentHead());
    float cpuct = ComputeCpuct(params_, node->GetN(), at_root);
    float puct_mult =
        cpuct * std::sqrt(std::max(node->GetChildrenVisits(), 1u));
    float fpu = GetFpu(params_, node, at_root, draw_score);
    m_evaluator.SetParent(node);

    classic::Node::Iterator best_edge_iter;
    float best_score = -std::numeric_limits<float>::infinity();

    // Edge examination limit: NStarted + cur_limit + 2 (matching search.cc).
    // In our single-visit walks, cur_limit=1 and we don't increment
    // n_in_flight during the walk, so we add 1 to compensate, giving N + 4.
    const int max_needed = std::min(static_cast<int>(node->GetNumEdges()),
                                    node->GetNStarted() + 4);
    int edges_examined = 0;

    for (auto edge : node->Edges()) {
      if (edges_examined >= max_needed) break;
      edges_examined++;

      int n = edge.GetNStarted();
      float p = edge.GetP();
      float q;
      if (n > 0) {
        q = edge.GetQ(fpu, draw_score);
        q += m_evaluator.GetMUtility(edge.node(), q);
      } else {
        q = fpu;
      }
      float score = q + p * puct_mult / (1 + n);
      if (score > best_score) {
        best_score = score;
        best_edge_iter = edge;
      }
    }

    if (!best_edge_iter) break;
    classic::Node* best_child = best_edge_iter.GetOrSpawnNode(node);
    classic::Edge* best_edge = best_edge_iter.edge();

    // TwoFold depth correction on tree reuse: if the selected child was
    // marked as a TwoFold terminal in a previous search but the repetition
    // cycle now extends before the current root, revert it.
    if (best_child->IsTwoFoldTerminal()) {
      const int depth = static_cast<int>(path.size());
      if (depth < best_child->GetM()) {
        const auto wl = best_child->GetWL();
        const auto d = best_child->GetD();
        const auto m = best_child->GetM();
        const auto terminal_visits = best_child->GetN();
        int depth_counter = 0;
        for (classic::Node* n = best_child; n != nullptr;
             n = n->GetParent()) {
          n->RevertTerminalVisits(wl, d, m + static_cast<float>(depth_counter),
                                  terminal_visits);
          depth_counter++;
          if (depth_counter > depth) break;
        }
        best_child->MakeNotTerminal();
      }
    }

    Move move = best_edge->GetMove();
    history.Append(move);
    node = best_child;
    path.push_back(node);
  }

  return node;
}


void BatchedSelfPlay::ProcessNNResult(LeafToEval& leaf) {
  // Set policies on the leaf's edges.
  size_t p_idx = 0;
  for (auto& edge : leaf.leaf->Edges()) {
    if (p_idx >= leaf.eval.p.size()) break;
    edge.edge()->SetP(leaf.eval.p[p_idx++]);
  }

  // Apply Dirichlet noise if enabled and this is the root.
  if (leaf.is_root && params_.GetNoiseEpsilon()) {
    ApplyDirichletNoise(leaf.leaf, params_.GetNoiseEpsilon(),
                        params_.GetNoiseAlpha());
  }

  leaf.leaf->SortEdges();
}

void BatchedSelfPlay::Backpropagate(
    const std::vector<classic::Node*>& path, float v, float d, float m) {
  for (int i = static_cast<int>(path.size()) - 1; i >= 0; i--) {
    // IncrementNInFlight to balance the decrement inside FinalizeScoreUpdate.
    path[i]->IncrementNInFlight(1);
    path[i]->FinalizeScoreUpdate(v, d, m, 1);
    v = -v;
    m = m + 1;
  }
}

void BatchedSelfPlay::MakeGameMove(GameState& game) {
  auto* root = game.tree->GetCurrentHead();
  const float draw_score = params_.GetDrawScore();
  const float fpu = GetFpu(params_, root, true, draw_score);

  // Find best move by visit count, breaking ties by Q then P
  // (matching GetBestChildrenNoTemperature in search.cc).
  classic::EdgeAndNode best_edge;
  uint32_t max_n = 0;
  for (auto edge : root->Edges()) {
    if (edge.GetN() > max_n ||
        (edge.GetN() == max_n && max_n > 0 &&
         (edge.GetQ(0.0f, draw_score) > best_edge.GetQ(0.0f, draw_score) ||
          (edge.GetQ(0.0f, draw_score) == best_edge.GetQ(0.0f, draw_score) &&
           edge.GetP() > best_edge.GetP())))) {
      max_n = edge.GetN();
      best_edge = edge;
    }
  }

  if (!best_edge) return;  // Should not happen.

  classic::Eval best_eval;
  best_eval.wl = best_edge.GetWL(-root->GetWL());
  best_eval.d = best_edge.GetD(root->GetD());
  best_eval.ml = best_edge.GetM(root->GetM() - 1) + 1;
  Move best_move = best_edge.GetMove();

  // Select played move with temperature.
  classic::EdgeAndNode played_edge;
  Move played_move;
  const int game_ply = game.tree->GetPositionHistory().Last().GetGamePly();
  float temperature = params_.GetTemperature();
  const int cutoff = params_.GetTemperatureCutoffMove();
  if (cutoff > 0 && game_ply >= cutoff * 2) {
    temperature = params_.GetTemperatureEndgame();
  }
  const int decay_delay = params_.GetTempDecayDelayMoves();
  const int decay_moves = params_.GetTempDecayMoves();
  if (decay_moves > 0 && game_ply >= decay_delay * 2) {
    int moves_since_delay = game_ply / 2 - decay_delay;
    if (moves_since_delay >= decay_moves) {
      temperature = 0.0f;
    } else {
      temperature *=
          static_cast<float>(decay_moves - moves_since_delay) / decay_moves;
    }
  }

  if (temperature < 0.01f) {
    // Effectively no temperature - use best move.
    played_edge = best_edge;
    played_move = best_move;
  } else {
    // Temperature-based sampling.
    const float offset = params_.GetTemperatureVisitOffset();
    float max_n_f = 0.0f;
    float max_eval = -1.0f;
    for (auto& edge : root->Edges()) {
      if (edge.GetN() + offset > max_n_f) {
        max_n_f = edge.GetN() + offset;
        max_eval = edge.GetQ(fpu, draw_score);
      }
    }
    const float min_eval =
        max_eval - params_.GetTemperatureWinpctCutoff() / 50.0f;

    std::vector<float> cumulative_sums;
    float sum = 0.0f;
    for (auto& edge : root->Edges()) {
      if (edge.GetQ(fpu, draw_score) < min_eval) continue;
      sum += std::pow(
          std::max(0.0f, (max_n_f <= 0.0f
                              ? edge.GetP()
                              : ((static_cast<float>(edge.GetN()) + offset) /
                                 max_n_f))),
          1.0f / temperature);
      cumulative_sums.push_back(sum);
    }

    if (cumulative_sums.empty() || sum <= 0.0f) {
      // Fallback to best move.
      played_edge = best_edge;
      played_move = best_move;
    } else {
      const float toss = Random::Get().GetFloat(cumulative_sums.back());
      int idx = std::lower_bound(cumulative_sums.begin(),
                                 cumulative_sums.end(), toss) -
                cumulative_sums.begin();
      for (auto& edge : root->Edges()) {
        if (edge.GetQ(fpu, draw_score) < min_eval) continue;
        if (idx-- == 0) {
          played_edge = edge;
          played_move = edge.GetMove();
          break;
        }
      }
    }
  }

  if (!played_edge) {
    played_edge = best_edge;
    played_move = best_move;
  }

  classic::Eval played_eval;
  played_eval.wl = played_edge.GetWL(-root->GetWL());
  played_eval.d = played_edge.GetD(root->GetD());
  played_eval.ml = played_edge.GetM(root->GetM() - 1) + 1;

  // Record training data.
  auto legal_moves =
      game.tree->GetPositionHistory().Last().GetBoard().GenerateLegalMoves();
  std::optional<EvalResult> nneval =
      player_.backend->GetCachedEvaluation(EvalPosition{
          game.tree->GetPositionHistory().GetPositions(), legal_moves});
  game.training_data.Add(root, game.tree->GetPositionHistory(), best_eval,
                         played_eval, false, best_move, played_move,
                         legal_moves, nneval,
                         params_.GetPolicySoftmaxTemp());

  game.move_count++;
  game.nodes_total += root->GetN();

  // Advance game. Edge moves are already in the mirrored (side-to-move at
  // bottom) format that NodeTree::MakeMove / PositionHistory::Append expects.
  game.tree->TrimTreeAtHead();
  game.tree->MakeMove(played_move);
}

void BatchedSelfPlay::CheckGameResult(GameState& game) {
  auto result = game.tree->GetPositionHistory().ComputeGameResult();
  if (result != GameResult::UNDECIDED) {
    game.result = result;
    return;
  }
  // 450-ply limit.
  if (game.tree->GetPositionHistory().Last().GetGamePly() >= 450) {
    game.result = GameResult::UNDECIDED;  // Adjudicated as draw-like.
    // Mark as finished by setting a non-UNDECIDED result.
    game.result = GameResult::DRAW;
    return;
  }
  // Syzygy tablebase probe.
  if (syzygy_tb_) {
    auto& board = game.tree->GetPositionHistory().Last().GetBoard();
    if (board.castlings().no_legal_castle() &&
        (board.ours() | board.theirs()).count() <=
            syzygy_tb_->max_cardinality()) {
      ProbeState state;
      const WDLScore wdl = syzygy_tb_->probe_wdl(
          game.tree->GetPositionHistory().Last(), &state);
      if (state != FAIL) {
        bool tb_side_black = (game.tree->GetPlyCount() % 2) == 1;
        if (wdl == WDL_WIN) {
          game.result = tb_side_black ? GameResult::BLACK_WON
                                      : GameResult::WHITE_WON;
        } else if (wdl == WDL_LOSS) {
          game.result = tb_side_black ? GameResult::WHITE_WON
                                      : GameResult::BLACK_WON;
        } else {
          game.result = GameResult::DRAW;
        }
      }
    }
  }
}

}  // namespace lczero
