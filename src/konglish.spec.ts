import { describe, expect, test } from "vitest";
import { Konglish } from "./konglish";

describe("Konglish smoke", () => {
  test("custom dictionary override", () => {
    const konglish = new Konglish({
      dictionary: { latte: ["라떼"], meetup: ["밋업"] },
    });

    const out = konglish.latinToHangul("latte meetup xylophone");
    expect(out.startsWith("라떼 밋업")).toBe(true);
  });
});
