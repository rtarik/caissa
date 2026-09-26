/** Messages between the page and the engine worker. */

export type ToEngine =
  | { kind: "load"; model: string; manifest: string }
  /**
   * `id` is echoed back with the answer. The page numbers every request and
   * keeps only the answer to the latest, so a search that finishes after an
   * undo or a new game - for a position that no longer exists - is dropped
   * rather than played into the wrong game.
   */
  | {
      kind: "move";
      id: number;
      moves: number[];
      simulations: number;
      /**
       * Where the moves start from, in the game's own notation, when it is not
       * the usual beginning: a FEN from the board editor. The engine replays the
       * moves from here instead; the network and the search are unchanged.
       */
      start?: string;
    };

export type FromEngine =
  | { kind: "ready"; game: string; generation?: number; parameters: number }
  | {
      kind: "move";
      id: number;
      action: number;
      value: number;
      /** Visit counts - or, when `simulations` is 0, the network's own priors. */
      visits: number[];
      simulations: number;
      ms: number;
    }
  | { kind: "error"; message: string; id?: number };
