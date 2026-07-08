# Project skills

The `gemini-*` skills here are copied verbatim from
[google-gemini/gemini-skills](https://github.com/google-gemini/gemini-skills),
licensed under Apache-2.0. They are kept locally so this project's Gemini work
(Live API voice, function calling, the Interactions API migration) has the
official guidance on hand.

Included:

- `gemini-live-api-dev` — Gemini Live API: streaming audio, VAD, native audio,
  function calling, session resumption. Directly matches Roberta's voice loop.
- `gemini-api-dev` — general `google-genai` SDK usage, model selection,
  multimodal input, structured output.
- `gemini-interactions-api` — the new Interactions API (successor to
  `generateContent`); relevant to the google-genai 2.x migration.

Note: these skills track the current Gemini generation and treat
`gemini-2.5-flash-native-audio-*` (what Roberta runs today) as legacy with a
migration path, so they may suggest model/config changes beyond a given task.
