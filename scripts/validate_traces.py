#!/usr/bin/env python3
"""Validate and print statistics for MCTS search trace binary files."""

import struct
import sys

MAGIC = 0x5443484D  # "MCHT"
VERSION = 1


def read_trace_file(filename):
    with open(filename, "rb") as f:
        # File header (16 bytes).
        magic, version, num_games, reserved = struct.unpack("<IIII", f.read(16))
        assert magic == MAGIC, f"Bad magic: 0x{magic:08X} (expected 0x{MAGIC:08X})"
        assert version == VERSION, f"Bad version: {version} (expected {VERSION})"
        print(f"Magic: 0x{magic:08X}  Version: {version}  Games: {num_games}")

        total_plies = 0
        total_traces = 0
        total_trace_moves = 0

        for gi in range(num_games):
            # Game header (4 bytes).
            num_plies, result, _ = struct.unpack("<HbB", f.read(4))
            result_str = {1: "white", -1: "black", 0: "draw"}.get(result, "?")

            # Game moves.
            played_moves = struct.unpack(f"<{num_plies}H", f.read(num_plies * 2))

            game_traces = 0
            game_trace_moves = 0
            ply_trace_counts = []

            for pi in range(num_plies):
                # Ply header (2 bytes).
                (num_traces,) = struct.unpack("<H", f.read(2))
                ply_trace_counts.append(num_traces)
                game_traces += num_traces

                for ti in range(num_traces):
                    (trace_len,) = struct.unpack("<B", f.read(1))
                    moves = struct.unpack(f"<{trace_len}H", f.read(trace_len * 2))
                    game_trace_moves += trace_len

            avg_depth = game_trace_moves / game_traces if game_traces else 0
            print(
                f"  Game {gi}: {num_plies} plies, {game_traces} traces, "
                f"avg depth {avg_depth:.1f}, result={result_str}"
            )

            total_plies += num_plies
            total_traces += game_traces
            total_trace_moves += game_trace_moves

        # Verify we're at EOF.
        remaining = f.read()
        if remaining:
            print(f"WARNING: {len(remaining)} bytes remaining after all games!")
        else:
            print("File structure OK (no trailing bytes).")

        print(f"\nTotals: {num_games} games, {total_plies} plies, "
              f"{total_traces} traces, {total_trace_moves} trace moves")
        if total_traces > 0:
            print(f"Avg traces/ply: {total_traces / total_plies:.1f}, "
                  f"avg depth: {total_trace_moves / total_traces:.1f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <trace_file.bin>")
        sys.exit(1)
    read_trace_file(sys.argv[1])
