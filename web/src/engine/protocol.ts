/** Messages between the page and the engine worker. */

export type ToEngine =
  | { kind: "load"; model: string; manifest: string }
  | { kind: "move"; moves: number[]; simulations: number };

export type FromEngine =
  | { kind: "ready"; game: string; generation?: number; parameters: number }
  | { kind: "move"; action: number; value: number; visits: number[]; ms: number }
  | { kind: "error"; message: string };
