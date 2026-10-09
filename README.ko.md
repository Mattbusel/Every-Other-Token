<p align="center">
  <a href="https://every-other-token.vercel.app/"><img src="assets/hero.svg" width="100%" alt="every-other-token이 답변을 토큰 단위로 스트리밍하는 모습: 한 토큰 걸러 하나씩 뒤집혀 강조 표시되고, 각 토큰 아래에 신뢰도 막대가 표시됨"></a>
</p>

<h1 align="center">every-other-token</h1>

<p align="center"><b>AI 모델이 쓰는 단어 하나하나를 얼마나 확신하는지 보고, 모델이 아직 쓰고 있는 동안 그 단어를 바꿔 보세요.</b></p>

<p align="center"><a href="README.md">English</a> | <a href="README.zh-CN.md">简体中文</a> | <a href="README.ja.md">日本語</a> | 한국어</p>

<p align="center">
  <a href="https://gitlab.com/mattbusel/Every-Other-Token/-/releases/permalink/latest/downloads/every-other-token-windows-x86_64.exe"><b>Windows용 다운로드 (.exe)</b></a> &nbsp;&middot;&nbsp;
  <a href="#install">Linux 및 macOS</a> &nbsp;&middot;&nbsp;
  <a href="https://every-other-token.vercel.app/">프로젝트 사이트</a> &nbsp;&middot;&nbsp;
  <a href="#documentation">문서</a>
</p>

<p align="center">
  <a href="https://crates.io/crates/every-other-token"><img src="https://img.shields.io/crates/v/every-other-token.svg?color=ff6a2b&labelColor=0c0d0b" alt="crates.io 버전"></a>
  <a href="https://docs.rs/every-other-token"><img src="https://img.shields.io/docsrs/every-other-token?labelColor=0c0d0b" alt="docs.rs"></a>
</p>

**설치도 API 키도 필요 없이 브라우저에서 바로 사용해 보세요:** [every-other-token.vercel.app/play](https://every-other-token.vercel.app/play/). 프롬프트를 입력하면 한 토큰 걸러 하나씩 뒤집혀 출력되는 것을 볼 수 있습니다.

`every-other-token`은 명령줄과 브라우저에서 쓸 수 있는 무료 LLM 토큰 스트림 뷰어이자 인터셉터입니다. OpenAI, Anthropic, Gemini, OpenRouter 또는 내 컴퓨터에서 돌아가는 모델(Ollama, llama.cpp, vLLM, LM Studio)의 실시간 스트리밍 출력에 끼어들어, logprobs를 바탕으로 각 토큰의 신뢰도(confidence)와 퍼플렉서티(perplexity)를 보여 주고, 토큰이 도착하는 대로 한 토큰 걸러 하나씩(또는 원하는 비율만큼) 다시 쓸 수 있습니다. 내장된 mock 프로바이더로 API 키 없이 모든 기능을 체험할 수 있습니다.

**이런 분께 추천합니다:** 언어 모델이 어떻게 단어를 고르는지 궁금한 모든 분, 그리고 LLM 해석 가능성(interpretability) 연구, 레드팀 테스트, 프롬프트 엔지니어링을 하는 분들.

## 작동 방식

<img src="docs/img/how-it-works.svg" width="100%" alt="실제 실행에서의 네 단계를 보여 주는 다이어그램: 1 Intercept가 실시간 스트림을 읽고, 2 Score가 각 토큰에 신뢰도 exp(logprob)와 퍼플렉서티 exp(-logprob)를 매기고, 3 Mutate가 홀수 번째 토큰을 뒤집고, 4 Output이 터미널, 웹 UI 또는 JSON에 'The kciuq brown xof jumps revo the yzal dog'를 표시함">

1. **가로채기(Intercept).** 모델과의 스트리밍 연결(SSE)을 직접 열기 때문에 각 청크가 도착하는 순간 바로 볼 수 있습니다.
2. **점수 매기기(Score).** 각 토큰에 `confidence = exp(logprob)`와 `perplexity = exp(-logprob)`, 그리고 모델이 고려했던 상위 후보들이 붙습니다.
3. **변형(Mutate).** 선택된 토큰(기본값은 한 토큰 걸러 하나, 또는 모델이 확신하지 못한 토큰만)에 변환을 적용합니다. 뒤집기, 대문자화, 노이즈, 삭제, 동의어 치환 등이 있습니다.
4. **출력(Output).** 터미널(일반 모드, 또는 `--tui`로 여는 전체 화면 보기)이나 웹 UI에서 지켜보거나, JSON Lines, CSV, HTML 히트맵으로 저장할 수 있습니다.

<a id="install"></a>
## 설치

**Linux** (x86_64, Ubuntu 20.04+ / Debian 11+). 의존성 없는 한 줄 명령으로 `~/.local/bin`에 설치됩니다:

```sh
mkdir -p ~/.local/bin && curl -fsSL https://gitlab.com/mattbusel/Every-Other-Token/-/releases/permalink/latest/downloads/every-other-token-linux-x86_64.tar.gz | tar xz --strip-components=1 -C ~/.local/bin --wildcards '*/every-other-token'
```

| 기타 운영체제 | |
|---|---|
| **Windows** | [every-other-token-windows-x86_64.exe 다운로드](https://gitlab.com/mattbusel/Every-Other-Token/-/releases/permalink/latest/downloads/every-other-token-windows-x86_64.exe) 후 실행하세요. (서명되지 않은 파일이라 SmartScreen이 확인을 요청할 수 있습니다. *추가 정보*를 누른 뒤 *실행*을 선택하세요.) |
| **macOS, 또는 소스에서 설치** | `cargo install --locked every-other-token` |

Windows .exe를 더블클릭하면 브라우저에서 웹 UI가 열립니다. SHA-256 체크섬이 포함된 모든 릴리스: [Releases](https://gitlab.com/mattbusel/Every-Other-Token/-/releases).

## 예제

아래는 모두 mock 프로바이더의 실제 출력입니다(고정된 답변과 고정된 logprobs를 실제 파이프라인에 통과시킨 것). 따라서 API 키 없이도 똑같이 재현할 수 있습니다.

**1. 한 토큰 걸러 하나씩 뒤집기** (기본 동작)

```console
$ every-other-token "Why is the sky blue?" --provider mock
The kciuq brown xof jumps revo the yzal dog. This si a kcom response rof prompt: Why si the yks blue?
24 tokens streamed, 12 transformed
```

**2. 모델이 확신하지 못한 단어만 다시 쓰기** (신뢰도 0.6 이하)

```console
$ every-other-token "Why is the sky blue?" uppercase --provider mock --rate 1 --min-confidence 0.6
The quick BROWN fox JUMPS over the LAZY dog. THIS is a mock response for PROMPT: WHY IS THE SKY BLUE?
24 tokens streamed, 11 transformed
```

"brown"(0.46), "jumps"(0.57), "lazy"(0.41), "This"(0.54), "prompt"(0.59)가 기준선 아래였습니다. 끝에 그대로 되풀이되는 프롬프트는 mock에서 logprobs가 없으므로 일반 비율(rate) 규칙으로 처리됩니다.

**3. 수치를 JSON으로 받기, 토큰마다 한 줄**

```console
$ every-other-token "Why is the sky blue?" uppercase --provider mock --json-stream
{"text":"The","original":"The","index":0,"transformed":false,"importance":0.8869204521179199,"confidence":0.88692045,"perplexity":1.1274968,"is_error":false,"arrival_ms":0}
{"text":" QUICK","original":" quick","index":1,"transformed":true,"importance":0.6376281380653381,"confidence":0.63762814,"perplexity":1.5683122,"is_error":false,"arrival_ms":0}
...
```

**4. 터미널에서 전체 화면으로 보기:** `every-other-token "Why is the sky blue?" --provider mock --tui`. 답변이 신뢰도에 따라 색이 입혀진 채로 스트리밍되고(초록은 확신, 노랑은 불확실, 빨강은 추측), 다시 쓴 토큰에는 밑줄이 그어지며, 옆 패널에는 실시간 통계, 신뢰도 스파크라인, 그리고 가장 최근 토큰에 대해 모델이 고려했던 다른 단어들이 표시됩니다. `q`를 누르면 종료되고, 답변은 터미널에 그대로 남습니다.

**5. 내 컴퓨터의 모델로 실행하기.** [Ollama](https://ollama.com)가 실행 중이라면 `every-other-token "Why is the sky blue?" --provider ollama`로 API 키 없이 `llama3.2`에서 스트리밍합니다. OpenAI API 호환 서버라면 무엇이든 동작합니다: `every-other-token "Why is the sky blue?" reverse my-model --base-url http://localhost:8080/v1`.

**6. 브라우저에서 보기:** `every-other-token --web --provider mock` (또는 .exe를 더블클릭). 분할 보기로 왼쪽에는 원본 스트림, 오른쪽에는 다시 쓴 스트림이 나오고, 각 토큰에는 신뢰도에 따른 밑줄이 그어집니다.

<img src="assets/web-ui.png" width="100%" alt="every-other-token 웹 UI 분할 보기: 왼쪽에 원본 토큰 스트림, 오른쪽에 변환된 스트림이 있고, 각 토큰에 신뢰도에 따른 밑줄이 그어져 있음">

## API 키 없이 정확한 확률 얻기: `--provider local`

호스팅 API는 토큰 확률을 아예 보내지 않거나(Anthropic, Gemini), 프롬프트의 어떤 단어가 중요했는지 알려 주지 못합니다. 도구 안에서 돌아가는 모델은 둘 다 할 수 있습니다. `local` 기능(feature)을 켜고 빌드하면 SmolLM2-135M-Instruct를 한 번만 내려받아(269 MB, 캐시 폴더에 저장) [candle](https://github.com/huggingface/candle)로 CPU에서 실행합니다:

```sh
cargo install every-other-token --features local
every-other-token "Reply with only the city name. Capital of France?" --provider local --attribute
```

실제 출력:

```text
The latipac of ecnarF is siraP.
How much the reply depended on each prompt word (drop in total log-probability when the word is removed):
    Reply   +1.432  #########
     with   +1.108  #######
     only   +1.875  ############
      the   +0.357  ##
     city   +3.499  ######################
    name.   +0.932  ######
  Capital   +3.229  #####################
       of   +0.857  #####
  France?   +4.689  ##############################
```

모든 토큰은 정확한 로그 확률(log-probability)과 모델의 실제 상위 5개 후보를 가지며, 터미널, `--tui`, 웹 UI, 모든 내보내기 형식에서 확인할 수 있습니다. `--attribute`는 프롬프트의 단어를 하나씩 빼고 같은 답변을 다시 채점해, 확률이 얼마나 떨어졌는지 보고합니다(가림 기반 기여도 분석, occlusion attribution). `--model`로 Hugging Face Hub의 다른 Llama 아키텍처 모델을 쓸 수 있고, `--seed N`을 주면 그리디 디코딩 대신 샘플링을 합니다.

확률을 돌려주지 않는 프로바이더(Anthropic, Gemini, 많은 로컬 서버)에서는 지어낸 신뢰도 대신 신뢰도를 아예 표시하지 않습니다.

## 3단계로 시작하기

1. **받기.** 위의 Linux 한 줄 명령이나 Windows .exe를 사용하세요(또는 `cargo install`).
2. **오프라인으로 체험하기.** `every-other-token "Why is the sky blue?" --provider mock`을 실행하거나, .exe를 더블클릭하고 프로바이더 메뉴에서 <b>Mock (no API key)</b>를 고르세요.
3. **실제 모델에 연결하기.** `OPENAI_API_KEY` 또는 `ANTHROPIC_API_KEY`(또는 `OPENROUTER_API_KEY`, `GEMINI_API_KEY`)를 설정한 다음 `every-other-token "Why is the sky blue?" --visual`(터미널, 신뢰도별 색상)을 실행하세요. 전체 화면으로 보려면 `--tui`를 추가하고, 브라우저로 보려면 `every-other-token --web`을 실행합니다. 프로바이더는 `--provider openai|anthropic|ollama|openrouter|gemini`로 고릅니다. 실제 신뢰도 수치는 logprobs에서 나옵니다(OpenAI, OpenRouter, 최신 Ollama). Anthropic이나 Gemini처럼 logprobs를 보내지 않는 프로바이더는 각 토큰이 얼마나 빨리 도착했는지로 추정값을 냅니다.

모든 플래그는 `every-other-token --help`로 볼 수 있고, 맨 아래에 예제가 있습니다.

<a id="documentation"></a>
## 문서

| 문서 | 내용 |
|---|---|
| [사용 가이드](docs/USAGE.md) | 모든 변환, `--rate`와 `--min-confidence`, 시스템 프롬프트 A/B 테스트, 프로바이더 간 비교, 연구 모드, 웹 UI 보기, 설정 파일, 셸 자동 완성, 소스에서 빌드 |
| [라이브러리 레퍼런스](docs/REFERENCE.md) | CLI를 이루는 Rust 모듈(기여도 내보내기, 드리프트 감지, 변형 실험실, 인과 맵 등)과 예제 |
| [아키텍처](docs/ARCHITECTURE.md) | 스트림, 변환, 출력이 어떻게 맞물리는지 |
| [HTTP API](docs/api.md) 및 [WebSocket 룸](docs/websocket.md) | 웹 서버 엔드포인트와 협업 토큰 편집 |
| [기능 플래그](docs/features.md) | 선택적 Cargo 기능(features) |
| [docs.rs](https://docs.rs/every-other-token) | 라이브러리로 쓸 때의 전체 API 문서 |
| [변경 이력](CHANGELOG.md) 및 [기여 가이드](CONTRIBUTING.md) | 릴리스 이력과 변경 사항을 보내는 방법 |

## 라이선스

MIT, [LICENSE](LICENSE)를 참고하세요.

## 개발 의뢰

**내 제품에도 이런 엔지니어링이 필요하신가요?** 소수의 클라이언트 프로젝트를 고정 가격으로 맡고 있습니다: LLM 기능, iOS 앱, 성능 개선. [서비스 및 가격](https://mattbusel.vercel.app/) · [이메일](mailto:mattbusel@gmail.com) · [LinkedIn](https://www.linkedin.com/in/matthewbusel/)
