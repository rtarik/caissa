/**
 * The four opponents the site offers.
 *
 * One dial: how many positions the engine looks at before it moves. The names
 * are what a player sees, and `scripts/levels.py` measures what each one is
 * actually worth in each game - the gaps run from about 380 Elo end to end in
 * Gomoku to nearly a thousand in Reversi.
 *
 * Beginner searches nothing at all and answers with the network's first
 * instinct. That is a real opponent rather than a hobbled one: it is what the
 * network knows without thinking, and it replies the instant you move.
 */
export interface Level {
  label: string;
  /** Positions searched per move; 0 plays straight from the network. */
  simulations: number;
  /** What that feels like, in a player's terms rather than an engine's. */
  note: string;
}

export const LEVELS: Level[] = [
  { label: "Beginner", simulations: 0, note: "answers instantly, without thinking ahead" },
  { label: "Casual", simulations: 40, note: "thinks a little before each move" },
  { label: "Strong", simulations: 200, note: "thinks properly; a real game" },
  { label: "Master", simulations: 600, note: "thinks hardest, and takes about a second" },
];

/**
 * The default is the strongest.
 *
 * Someone who wants an easier game will go looking for the setting; someone who
 * does not will otherwise never find out how well the thing plays. Master takes
 * under a second a move in every game here, so there is nothing to protect them
 * from - measured at 898 ms for chess and 990 ms for Dots & Boxes.
 */
export const DEFAULT_LEVEL = LEVELS.length - 1;
