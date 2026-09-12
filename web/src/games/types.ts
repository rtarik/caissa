/**
 * The browser's half of the `Game` contract.
 *
 * Deliberately the same shape as `src/caissa/games/base.py`, including the two
 * conventions that everything else depends on: positions are always described
 * from the point of view of the player about to move, and values are always that
 * player's result. Diverge from either and the network receives inputs it was
 * never trained on - while continuing to produce legal, plausible moves.
 */
export interface Game<S> {
  readonly name: string;
  /** Number of distinct actions; the network's policy head emits this many. */
  readonly actionSize: number;
  /** [height, width] of the board. */
  readonly boardShape: readonly [number, number];
  /** Feature planes produced by {@link encode}. */
  readonly inputPlanes: number;

  initialState(): S;
  /** One entry per action, true where the move is legal. */
  legalActions(state: S): boolean[];
  apply(state: S, action: number): S;
  /** `null` while the game continues, otherwise the result for the mover. */
  terminalValue(state: S): number | null;
  /** Network input: `inputPlanes x height x width`, flattened, canonical. */
  encode(state: S): Float32Array;
}
