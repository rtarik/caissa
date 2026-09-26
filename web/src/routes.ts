/**
 * Which screen the URL is asking for.
 *
 * Kept apart from `main.ts`, which starts a worker and rewrites the document the
 * moment it is imported, so the rules about what a URL means can be read - and
 * tested - on their own.
 */
import { LADDER } from "./games/ladder";

export type Route =
  | { name: "gallery" }
  | { name: "play"; key: string }
  | { name: "games"; key: string }
  | { name: "about" };

/** The link that reaches a screen. */
export function href(route: Route): string {
  if (route.name === "about") return "#/how-it-works";
  if (route.name === "play") return `#/play/${route.key}`;
  if (route.name === "games") return `#/games/${route.key}`;
  return "#/";
}

/**
 * Anything unrecognised is the gallery, deliberately.
 *
 * A stale bookmark, a game that has been renamed, a hand-typed hash: all of them
 * land somewhere that works rather than on an empty screen or an error.
 */
export function parseRoute(hash: string): Route {
  const path = hash.replace(/^#\/?/, "");
  if (path === "how-it-works") return { name: "about" };
  for (const name of ["play", "games"] as const) {
    if (!path.startsWith(`${name}/`)) continue;
    const key = path.slice(name.length + 1);
    if (LADDER.some((item) => item.key === key)) return { name, key };
  }
  return { name: "gallery" };
}
