import { describe, expect, it } from "vitest";
import { networksByGame } from "../src/models";

describe("the networks on offer", () => {
  it("groups by game, newest first, and names what has no label", () => {
    const networks = networksByGame([
      { game: "chess", file: "chess", generation: 0 },
      { game: "chess", file: "chess-imitation1", label: "Imitation 1", generation: 1 },
      { game: "connect4", generation: 40 },
      { game: "chess", file: "chess-imitation2", label: "Imitation 2", generation: 2 },
    ]);

    expect(networks.get("chess")!.map((n) => [n.file, n.label])).toEqual([
      ["chess-imitation2", "Imitation 2"],
      ["chess-imitation1", "Imitation 1"],
      ["chess", "Untrained"],
    ]);
    // An index written before files were named: the file is the game's own.
    expect(networks.get("connect4")).toEqual([
      { game: "connect4", file: "connect4", label: "Generation 40", generation: 40 },
    ]);
  });
});
