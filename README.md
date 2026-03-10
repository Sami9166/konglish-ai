# Konglish-AI

영어/외래어 문장을 한국어 발음 표기로 바꿔주는 TypeScript 라이브러리입니다.

기본적으로 커스텀 사전을 사용해 단어를 한국어 발음으로 변환합니다.  
영어는 철자만으로 발음을 정확히 판단하기 어려우므로, 자주 쓰는 단어와 표현은 사전에 직접 등록해서 사용하는 방식을 중심으로 설계했습니다.

- 동기 함수는 사전 기반으로만 동작합니다. 사전에 없는 단어를 만났을 시 원문을 반환합니다.
- 비동기 함수는 사전에 없는 단어를 만났을 시 fallback 방식으로 Seq2Seq 모델을 사용합니다.

사전에 없는 단어를 어떻게 처리할지에 따라 동기/비동기 API를 선택해서 사용할 수 있습니다.

## 설치

```bash
pnpm add konglish-ai
# 또는
npm install konglish-ai
# 또는
yarn add konglish-ai
```

## 사용법

### 동기 함수

```ts
import { latinToHangul } from "konglish-ai";

latinToHangul("good morning"); // "굿 모닝"
latinToHangul("coffee time"); // "커피 타임"
latinToHangul("family vacation"); // "패밀리 베케이션"
latinToHangul("weekend movie night"); // "위켄드 무비 나이트"
latinToHangul("happy birthday"); // "해피 버스데이"
latinToHangul("pizza party"); // "피자 파티"
```
사전에 없는 단어가 포함되어 있다면:
```ts
latinToHangul("latte meetup zylphora"); // 라떼 밋업 zylphora
```
### 비동기 함수
```ts
import { latinToHangulAsync } from "konglish-ai";

await latinToHangulAsync("good morning"); // "굿 모닝"
await latinToHangulAsync("coffee time"); // "커피 타임"
await latinToHangulAsync("family vacation"); // "패밀리 베케이션"
await latinToHangulAsync("weekend movie night"); // "위켄드 무비 나이트"
await latinToHangulAsync("happy birthday"); // "해피 버스데이"
await latinToHangulAsync("pizza party"); // "피자 파티"
```
사전에 없는 단어가 포함되어 있다면:
```ts
await latinToHangulAsync("latte meetup zylphora"); // 라떼 밋업 질포라
```
비동기 함수는 사전 기반으로 변환을 시도하고 fallback이 일어나면 Seq2Seq 모델로 발음을 추론합니다.

`latinToHangulAsync`는 Promise를 반환하므로 비동기 코드에서 `await`로 호출할 수 있습니다.

### 클래스 호출

```ts
import { Konglish } from "konglish-ai";

const konglish = new Konglish({
  dictionary: {
    latte: ["라떼"],
    meetup: ["밋업"],
  },
});

const sync = konglish.latinToHangul("latte meetup"); // "라떼 밋업"
const asyncOut = await konglish.latinToHangulAsync(
  "latte meetup zylphora",
); // "라떼 밋업 질포라"
```

## 모델 및 데이터셋
자세한 내용은 아래 문서를 참고하시길 바랍니다.
- [모델 및 데이터셋 설명](./docs/model.md)

## 라이선스

[MIT](./LICENSE)