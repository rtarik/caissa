/**
 * Guards on the one thing a stylesheet can break that no other test sees.
 *
 * Board cells are buttons, and a CSS rule can make a button unclickable without
 * changing a line of logic: every unit test still passes, the markup is right,
 * the class is right, and the click does nothing. That happened - chess arrived
 * with a global `.piece` rule carrying `pointer-events: none`, Isola had been
 * using `piece` as a cell class since long before, and the square Isola most
 * needs you to click (the one your piece just left) went dead.
 *
 * So: class names that mean different things in different games must not be
 * spelled the same, and no unscoped rule may disable pointing at a cell.
 */
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { pieceSvg } from "../src/ui/pieces";

const css = readFileSync(new URL("../src/style.css", import.meta.url), "utf8");

/** Selectors of every rule in the sheet, one per entry. */
const selectors = css
  .replace(/\/\*[\s\S]*?\*\//g, "")
  .split("}")
  .map((block) => block.split("{")[0].trim())
  .filter((selector) => selector.length > 0 && !selector.startsWith("@"));

describe("the stylesheet", () => {
  it("keeps the chess pieces in a namespace of their own", () => {
    const svg = pieceSvg("p", "w");
    expect(svg).toContain("chess-piece");
    expect(svg).not.toMatch(/class="piece[ "]/);
  });

  it("has no unscoped rule for a class the boards use as a cell class", () => {
    // Shared between games: `piece`, `you`, `engine`, `win`, `last` and the rest
    // all appear on cells, and each board means something different by them.
    const shared = ["piece", "you", "engine", "win", "last", "ghost", "stone"];
    const unscoped = selectors.filter((selector) =>
      selector
        .split(",")
        .some((part) => shared.some((name) => part.trim().startsWith(`.${name}`))),
    );

    // `.stone` is the exception, and deliberately so: it is one element drawn
    // the same way wherever it appears, never a cell.
    expect(unscoped.filter((s) => !s.startsWith(".stone"))).toEqual([]);
  });

  it("never turns off pointer events on something a player has to click", () => {
    const blocks = css.split("}");
    const deadly = blocks
      .filter((block) => /pointer-events:\s*none/.test(block))
      .map((block) => block.split("{")[0].trim());

    // Pseudo-elements and drawn overlays may ignore the mouse; a cell may not.
    // `.token` is Isola's pawn, drawn *inside* the button it stands on - the
    // click has to reach the button, so the drawing must not intercept it.
    for (const selector of deadly) {
      expect(selector).toMatch(/::(after|before)|\.stone|\.chess-piece|\.token|\.coord/);
    }
  });
});
