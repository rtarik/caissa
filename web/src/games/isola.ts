import type { Game } from "./types";

export const SIZE = 7;
export const SQUARES = SIZE * SIZE;

/**
 * The eight compass directions, in a fixed order.
 *
 * The order is part of the action encoding, so it must match
 * `DIRECTIONS` in `src/caissa/games/isola.py` exactly and must not change once
 * a network has been trained against it.
 */
export const DIRECTIONS: ReadonlyArray<readonly [number, number]> = [
  [-1, -1], [-1, 0], [-1, 1],
  [0, -1], [0, 1],
  [1, -1], [1, 0], [1, 1],
];

/** Eight directions times forty-nine squares - a compound action, multiplied. */
export const ACTIONS = DIRECTIONS.length * SQUARES;

export interface IsolaState {
  /** Squares still standing. */
  readonly usable: Uint8Array;
  /** Flat index of the player about to move. */
  readonly mover: number;
  readonly opponent: number;
  readonly ply: number;
}

/** The square one step from `square`, or -1 if it leaves the board. */
export function step(square: number, direction: number): number {
  const row = Math.floor(square / SIZE) + DIRECTIONS[direction][0];
  const col = (square % SIZE) + DIRECTIONS[direction][1];
  if (row < 0 || row >= SIZE || col < 0 || col >= SIZE) return -1;
  return row * SIZE + col;
}

export class Isola implements Game<IsolaState> {
  readonly name = "isola";
  readonly actionSize = ACTIONS;
  readonly boardShape = [SIZE, SIZE] as const;
  /** The mover's piece, the opponent's piece, and the squares still standing. */
  readonly inputPlanes = 3;

  initialState(): IsolaState {
    return {
      usable: new Uint8Array(SQUARES).fill(1),
      mover: Math.floor(SIZE / 2),
      opponent: SQUARES - 1 - Math.floor(SIZE / 2),
      ply: 0,
    };
  }

  legalActions(state: IsolaState): boolean[] {
    const legal = new Array<boolean>(ACTIONS).fill(false);
    for (let direction = 0; direction < DIRECTIONS.length; direction++) {
      const target = step(state.mover, direction);
      if (target < 0 || !state.usable[target] || target === state.opponent) continue;

      // Anything still standing except the two occupied squares - including the
      // square just vacated, which is free again.
      const base = direction * SQUARES;
      for (let square = 0; square < SQUARES; square++) {
        if (!state.usable[square]) continue;
        if (square === target || square === state.opponent) continue;
        legal[base + square] = true;
      }
    }
    return legal;
  }

  apply(state: IsolaState, action: number): IsolaState {
    if (!this.legalActions(state)[action]) {
      throw new Error(`illegal action ${action}`);
    }
    const direction = Math.floor(action / SQUARES);
    const destroy = action % SQUARES;
    const target = step(state.mover, direction);

    const usable = new Uint8Array(state.usable);
    usable[destroy] = 0;

    // The mover becomes the opponent: one swap per apply, so the perspective
    // stays canonical exactly as the board flips sign in the other games.
    return { usable, mover: state.opponent, opponent: target, ply: state.ply + 1 };
  }

  terminalValue(state: IsolaState): number | null {
    // A player with no legal action has lost, and there are no draws. The
    // opposite of Reversi, where having no move means pass.
    for (let direction = 0; direction < DIRECTIONS.length; direction++) {
      const target = step(state.mover, direction);
      if (target < 0 || !state.usable[target] || target === state.opponent) continue;
      // A legal step always has a legal demolition: the vacated square is
      // always standing and never occupied afterwards.
      return null;
    }
    return -1;
  }

  encode(state: IsolaState): Float32Array {
    const planes = new Float32Array(this.inputPlanes * SQUARES);
    planes[state.mover] = 1;
    planes[SQUARES + state.opponent] = 1;
    for (let i = 0; i < SQUARES; i++) {
      if (state.usable[i]) planes[2 * SQUARES + i] = 1;
    }
    return planes;
  }

  /** Squares the mover can step to, for the UI. */
  steps(state: IsolaState): number[] {
    const targets: number[] = [];
    for (let direction = 0; direction < DIRECTIONS.length; direction++) {
      const target = step(state.mover, direction);
      if (target >= 0 && state.usable[target] && target !== state.opponent) {
        targets.push(target);
      }
    }
    return targets;
  }

  /** The direction index from `from` to `to`, or -1 if they are not adjacent. */
  directionBetween(from: number, to: number): number {
    for (let direction = 0; direction < DIRECTIONS.length; direction++) {
      if (step(from, direction) === to) return direction;
    }
    return -1;
  }
}
