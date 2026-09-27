import { describe, expect, it } from "vitest";
import { networkByGame } from "../src/models";

describe("the network each game plays", () => {
  it("is the newest one the index lists, wherever it sits in the list", () => {
    const networks = networkByGame([
      { game: "chess", file: "chess-imitation1", generation: 1 },
      { game: "chess", file: "chess-imitation2", generation: 2 },
      { game: "connect4", generation: 40 },
      { game: "chess", file: "chess", generation: 0 },
    ]);
    expect(networks.get("chess")).toEqual({ game: "chess", file: "chess-imitation2", generation: 2 });
  });

  it("is the game's own file in an index written before files were named", () => {
    expect(networkByGame([{ game: "connect4", generation: 40 }]).get("connect4"))
      .toEqual({ game: "connect4", file: "connect4", generation: 40 });
  });
});
