/**
 * The games, in the order they are being built.
 *
 * Each one adds exactly one new difficulty to the framework, which is why the
 * order matters and why the menu shows all of them rather than only the ones
 * that are finished - the sequence is part of what the project is.
 *
 * Availability is not declared here. It comes from `models/index.json`, written
 * by the export script from the files that actually exist, so the menu cannot
 * offer a game whose network is missing or hide one that has just been trained.
 *
 * `title` is display text and `key` is the identifier used for model files,
 * checkpoints and registry lookups. They differ on purpose for two games:
 * "Connect 4" is a live Hasbro trademark and "Isola" is Ravensburger's, so the
 * page shows the generic names while the keys - and everything already trained
 * against them - stay put. Reversi, Gomoku and Dots & Boxes need no such care;
 * those are the unencumbered names already (it is *Othello* that is the
 * trademark, which is why nearly all software says Reversi).
 */
export interface LadderEntry {
  key: string;
  title: string;
  /** What the game is, in a few words. */
  blurb: string;
  /** How to win, for someone who has never played it. Shown while playing. */
  howToWin: string;
  /** What it forces the framework to handle that earlier games did not.
   *  Kept off the play screen: it is about the project, not about the game. */
  teaches: string;
  /** Names for the two seats, where the game has its own: chess's White and Black. */
  seats?: [string, string];
}

export const LADDER: LadderEntry[] = [
  {
    key: "connect4",
    title: "Four in a Row",
    blurb: "Drop a disc; gravity does the rest.",
    howToWin: "Get four of your discs in a line - across, up or diagonally.",
    teaches: "The baseline. Solved, so play can be graded against perfect.",
  },
  {
    key: "reversi",
    title: "Reversi",
    blurb: "Flank a line of discs to flip it.",
    howToWin: "Trap a line of your opponent's discs between two of yours to flip them. Most discs at the end wins.",
    teaches: "Passing: having no legal move does not end the game.",
  },
  {
    key: "gomoku",
    title: "Gomoku",
    blurb: "Five in a row on an open board.",
    howToWin: "Place a stone anywhere. Five in a row - across, up or diagonally - wins.",
    teaches: "A large action space, and very sparse policy targets.",
  },
  {
    key: "isola",
    title: "Isolation",
    blurb: "Move your piece, then destroy a square.",
    howToWin: "Each turn, step one square and then destroy any empty square. Strand your opponent with nowhere to step.",
    teaches: "Compound actions — a rehearsal for chess's move encoding.",
  },
  {
    key: "dotsandboxes",
    title: "Dots & Boxes",
    blurb: "Draw lines; close a box to claim it — and move again.",
    howToWin: "Draw one line per turn. Complete the fourth side of a box to claim it and go again. Most boxes wins.",
    teaches: "Turns that don't always pass, so the framework must be told whose move it is.",
  },
  {
    key: "chess",
    title: "Chess",
    blurb: "The main event.",
    howToWin: "Checkmate the enemy king. All the usual rules, including castling, en passant and promotion.",
    teaches: "Everything at once, plus a bootstrap from human games.",
    seats: ["White", "Black"],
  },
];

export const TITLES: Record<string, string> = Object.fromEntries(
  LADDER.map((entry) => [entry.key, entry.title]),
);
