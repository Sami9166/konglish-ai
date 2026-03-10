import { describe, expect, it } from "vitest";
import { latinToHangul, latinToHangulAsync } from "./latinTokorea";

// pnpm test latinTokorea.spec.ts

describe("latinToHangul", () => {
  it('pretender를 "프리텐더"로 변환해야 함', () => {
    expect(latinToHangul("pretender")).toBe("프리텐더");
  });

  it("특수문자와 공백을 보존해야 함", () => {
    expect(latinToHangul("(pretender!)")).toBe("(프리텐더!)");
  });

  it("사전 override 옵션을 적용할 수 있어야 함", () => {
    expect(latinToHangul("codex", { dictionary: { codex: ["코덱스"] } })).toBe(
      "코덱스",
    );
  });

  it("README 예제를 그대로 재현해야 함", async() => {
    const cases: Array<[string, string]> = [
      ["good morning", "굿모닝"],
      ["coffee time", "커피 타임"],
      ["family vacation", "패밀리 베케이션"],
      ["weekend movie night", "위켄드 무비 나이트"],
      ["happy birthday", "해피 버스데이"],
      ["pizza party", "피자 파티"],
      ["music festival", "뮤직 페스티벌"],
      ["new project", "뉴 프로젝트"],
      ["thank you", "땡큐"],
      ["zylphora", "질포라"],   // AI 추론용 단어
    ];

    for (const [input, expectedOutput] of cases) {
      await expect(
        latinToHangulAsync(input)
      ).resolves.toBe(expectedOutput);
    }
  });
});
