Reference sources:
- Google AI Studio TTS prompt guide with tags: https://aistudio.google.com/learn/gemini-tts-prompt-guide-with-tags
- Gemini API text-to-speech docs, "Controlling speech style with prompts" and "Audio tags": https://ai.google.dev/gemini-api/docs/speech-generation

# Practical Gemini TTS audio tags

Gemini TTS can be steered with natural-language prompting and inline audio tags. Google describes audio tags as bracketed inline modifiers, such as `[whispers]` or `[laughs]`, that can change tone, pace, emotion, or add non-verbal sounds. Google also notes that there is no exhaustive guaranteed list of working tags, so the safest practice is to use common, documented examples first and test persona-specific tags in AI Studio.

If the transcript is not in English, Google recommends keeping the audio tags in English.

## Good default tags to try

Use these first when writing persona templates or speech examples:

| Purpose | Tags |
| --- | --- |
| Pauses and timing | `[short pause]`, `[long pause]` |
| Quiet delivery | `[whispers]` |
| Loud delivery | `[shouting]` |
| Laughter | `[laughs]`, `[giggles]` |
| Breath or reaction sounds | `[sighs]`, `[gasp]` |
| Energy and affect | `[excited]`, `[amazed]`, `[curious]`, `[serious]`, `[tired]` |
| Negative or unstable affect | `[crying]`, `[panicked]`, `[trembling]` |
| Attitude | `[sarcastic]`, `[mischievously]`, `[reluctantly]`, `[bored]` |
| Pace | `[very fast]`, `[very slow]` |

## Spellings to prefer

Prefer the spellings shown in Google's examples:

- Use `[whispers]`, not `[whispering]`, for whispering.
- Use `[laughs]` or `[giggles]`, not `[laughing]`, for laughter.
- Use `[sighs]`, not `[sigh]`, for sighs.
- Use `[sarcastic]` or `[sarcastically]`, not `[sarcasm]`, for sarcasm.
- Use `[excited]` or `[excitedly]`, not `[excitement]`, for excited delivery.

## Placement patterns

Put a tag at the start of a line or sentence when it should affect the whole utterance:

```text
[excited] Motion complete. Awaiting the next command.
```

Put a tag inline when only the following phrase should change:

```text
Status is nominal. [whispers] Quiet mode is active.
```

Use tags sparingly. A short spoken reply usually needs zero or one tag. Too many tags can make the output sound over-directed or cause the tag text to be spoken literally.

## Persona examples

Bobby, gentle and soft:

```text
Of course. [short pause] I will move carefully.
```

```text
[whispers] I am here if you need me.
```

Cyclop, mechanical and curt:

```text
[serious] Target required.
```

```text
[sarcastic] Compliment logged. Productivity unchanged.
```

Kibbitz, dry and theatrical:

```text
[sighs] Hmmmmmm. Motion confirmed.
```

```text
[very slow] A flawless maneuver, naturally.
```

## Notes for this project

- Keep speech tags in final spoken text only.
- Do not put speech tags in tool calls, coordinates, JSON, plan names, or internal reasoning.
- For robot operation, tool rules and observed state always override voice style.
- For production personas, prefer a small local allowlist of tested tags rather than relying on every creative tag that may work in AI Studio.

# Existing notes

Gemini 3.1 flash TTS supports 200+ audio tags to prompt expressive voices.

Most commonly used tags include: [determination], [enthusiasm], [adoration], [interest], [awe], [admiration], [nervousness], [frustration], [excitement], [curiosity], [hope], [annoyance], [amusement], [aggression], [tension], [agitation], [confusion], [anger], [positive], [neutral], [negative], [whispers], and [laughs].

Pacing and stylistic controls: you can use pacing tags like [slow] or [fast] to control the speed of the delivery. To pace out your information and let dramatic moments land, use tags like [short pause] or [long pause].

Non-verbal vocalizations: the model can produce realistic non-verbal audio. You can insert tags like [laughs] or [whispers] to add texture to the audio output.
