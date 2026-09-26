/**
 * The browser's half of the `Game` contract.
 *
 * Deliberately the same shape as `src/caissa/games/base.py`, including the
 * conventions that everything else depends on: positions are always described
 * from the point of view of the player about to move, values are always that
 * player's result, and whose turn it is is stated by the game rather than
 * worked out by counting moves. Diverge from any of them and the network receives
 * inputs it was never trained on - while continuing to produce legal, plausible
 * moves.
 */
export interface Game<S> {
  readonly name: string;
  /** Number of distinct actions; the network's policy head emits this many. */
  readonly actionSize: number;
  /** [height, width] of the board. */
  readonly boardShape: readonly [number, number];
  /** Feature planes produced by {@link encode}. */
  readonly inputPlanes: number;
  /**
   * The action that means "pass", for games that have one - Reversi does.
   *
   * Declared rather than guessed. The page used to assume a pass was the last
   * action index, which is true for Reversi and false everywhere else: in Four in
   * a Row, with only the seventh column open, it dropped the disc for you and
   * announced that you had passed.
   */
  readonly passAction?: number;

  initialState(): S;
  /**
   * A position described in the game's own notation, for games that can start
   * somewhere other than the beginning - a FEN, for chess. Optional: only chess
   * has a board editor. Nothing about the network or the search changes for a
   * position reached this way; they already play whatever position they are
   * given. What changes is only where the replay of the moves begins.
   */
  positionFrom?(description: string): S;
  /**
   * Which seat is to move: 0 for whoever moved first, 1 for the other.
   *
   * Mirrors `to_play` in `src/caissa/games/base.py`. Canonical perspective hides
   * it, so it has to be stated: in Dots & Boxes a closed box earns another move,
   * and counting moves would hand that bonus move to the wrong player.
   */
  toPlay(state: S): number;
  /** One entry per action, true where the move is legal. */
  legalActions(state: S): boolean[];
  apply(state: S, action: number): S;
  /** `null` while the game continues, otherwise the result for the mover. */
  terminalValue(state: S): number | null;
  /** Network input: `inputPlanes x height x width`, flattened, canonical. */
  encode(state: S): Float32Array;
}
