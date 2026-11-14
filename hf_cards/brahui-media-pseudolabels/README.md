---
language:
- brh
license: other
license_name: research-only-transcripts
task_categories:
- automatic-speech-recognition
tags:
- brahui
- pseudo-labels
- machine-generated
- low-resource
size_categories:
- 10K<n<100K
configs:
- config_name: default
  data_files: train.jsonl
---

# Brahui broadcast speech: machine transcripts with timestamps (no audio)

Automatic transcripts of 160 hours of Brahui speech from 1,369 public YouTube videos: news bulletins, talk shows,
stories, lessons and drama. Each row gives a video ID, a start and end time, and the text a speech recognizer
produced for that span.

**There is no audio in this dataset.** The recordings belong to their broadcasters and uploaders. To use the audio
you fetch it yourself from the original videos, under whatever terms apply to you.

**The text is machine output, not human transcription.** It was produced by this project's fine-tuned Brahui
recognizer ([qkhawaja/brahui-asr-ctc1b-s3](https://huggingface.co/qkhawaja/brahui-asr-ctc1b-s3)), which makes
about 9 character errors per 100 on clean read speech and more on broadcast audio with music, noise or several
speakers. Nobody has corrected it.

## Contents

45,749 segments, 160.5 hours.

| kind | hours | videos |
|---|---|---|
| Daily news bulletins (Balochistan's Today) | 50.9 | 537 |
| Mixed programming (Brahvi Tawar Media, Mehr TV, Brahui TV) | 52.3 | 415 |
| Stories and narration | 21.2 | 266 |
| Talk shows, interviews, current affairs (incl. PTV Bolan) | 21.0 | 59 |
| Language lessons | 9.9 | 70 |
| Drama, sermons, poetry, vlogs | 5.1 | 22 |

Fields: `video_id`, `url`, `collection`, `channel`, `upload_date`, `segment`, `start`, `end` (seconds from the
start of the video), `text`, `confidence` (mean per-frame probability of the recognizer's output; median 0.92, and
a tenth of segments are below 0.80).

## What it is good for

- Pseudo-labels for training speech recognizers, after filtering by `confidence`. In this project, 10 hours of
  these labels helped train a 300M model that runs in the browser.
- Text for language models. A character 6-gram model built partly on this text cut a recognizer's error rate
  from 9.9 to 8.0.
- Finding Brahui speech of a particular kind to listen to.

Videos may have been removed since collection. Some videos mix Brahui with Urdu, Balochi or English, and the
recognizer writes all of it as if it were Brahui.

## Terms

The transcripts are offered for research and for building language technology for Brahui. They are derived from
other people's broadcasts, so treat them the way you would treat quotations from those broadcasts. Recitations of
a published Quran translation and a film dub were collected for the project but are left out here.

If you made or appear in one of these recordings and want its rows removed, open a discussion in the Community tab
and they will be removed.

Method, code and results: https://github.com/qasimy123/brahui-digitize
