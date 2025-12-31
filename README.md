# Mac video text extractor
A Python script that extracts burned-in subtitle from video file on Mac.
Uses the vision framework from Apple.

It takes a video file, decode the frames, and extracts the texts before writing to a subtitle file.

I wrote this as doing OCR on videos on mac is too slow using cross-platform tools. This is a personal project, so not expect quick fixes or response to issues. Make your own patch if you know how to or open an issue/pull.

---

## Requirements

### Python
- Python **3.10+** is required.

### Third-Party CLI Tools
- **ffmpeg** must be installed and available in PATH:
```bash
brew install ffmpeg
```

### Python Libraries

Install required Python packages via pip3:

```bash
pip3 install ocrmac pillow tqdm numpy
```

# Usage

## 1. Run as a Script (CLI)
Run the script with Python 3.10+:
```bash
python3 main.py [options]
```

### Command-Line Arguments
| Argument | Type | Description | Example |
|----------|------|-------------|---------|
| `input_path` | positional | 1 or more paths to video to process | `python3 main.py video.mp4` |
| `--pattern` | optional | Regex select text and str to replace them with. The replacement str defaults to empty string if not provided. | `python3 main.py video.mp4 --pattern 1 I` |
| `--lyrics` | optional | A file of text to match extracted text to. The texts are chronological, separated by at least 2 line, and will not be changed by ``--pattern``. Each file passed only applies to 1 input video, any more or less will not be applied. | `python3 main.py video.mp4 --lyrics lyrics_file.txt` |
| `--min-change` | optional | The minium change between 2 frames before a new frame is extracted. Accepted values are from 0-1. Defaults is 0.02. Use 0 to extract all frames | `python3 main.py video.mp4 --min-change 0.1` |
| `--crop` | optional | The box to crop the video to. This value is passed to ffmpeg. The syntax is `width:height:x_pos:y_pos`. | `python3 main.py video.mp4 --crop in_w/2:in_h/2:in_w/4:in_h/4` |
| `--lang` | optional | The languages used by the OCR. Defaults to None | `python3 main.py video.mp4 --lang en-US  de-DE` |
| `--subtract-from-pre` | optional | Seconds to subtract from each cue. Defaults to 0. | `python3 main.py video.mp4 --subtract-from-pre 0.1` |
| `--min-seg-len-for-subtract` | optional | The minimum length in seconds of each cue to apply `--subtract-from-pre`. Defaults to 0. | `python3 main.py video.mp4 --subtract-from-pre 0.1 --min-seg-len-for-subtract 0.15` |
| `--version` | optional | The version to use. Version 2 is faster, version 1 is lighter on resources but very slow and cannot adjust output format. Defaults to 2. | `python3 main.py video.mp4 --format 1` |
| `--format` | optional | The format used to save the output. Accepts `ass`, `srt`, `vtt`, and `webvtt`. Defaults to `vtt`.| `python3 main.py video.mp4 --format srt` |

### Examples

#### Process a single video
```bash
python3 main.py v.mp4 --lang en-US --subtract-from-pre 0.1 --min-seg-len-for-subtract 0.15 --pattern 1 i --lyrics lyrics.txt --crop in_w:130:0:230 --min-change 0
```

#### Process multiple videos at once
```bash
python3 main.py v1.mp4 v2.mkv v3.webm v4.mov --lang en-US --subtract-from-pre 0.1 --min-seg-len-for-subtract 0.15 --pattern 1 i --lyrics lyrics1.txt lyrics1.txt lyrics2.txt --crop in_w:130:0:230 --min-change 0
```

### 2. Import as a Python Module

You can also use your script as a module and call its main function(s) directly from other Python code.
```python
video2vtt(
    video_path: str,
    remove: str = "",
    replace: str = "",
    scene: float = 0.02,
    lyrics_file: TextIO | None = None,
    sub_from_prev: float = 0,
    min_to_sub: float = 0,
    crop: str = "in_w:in_h:0:0",
    lang: list[str] | None = None,
    write_file: bool = True,
    out_format: str = "vtt",
)
```
Arguments:
| Name | Type | Information | Default |
| --- | --- | --- | --- |
| `video_path` | `str` | The path to 1 video file | This is required |
| `remove` | `str` | Regex to select texts to remove | `` |
| `replace` | `str` | String to replace selected text from above | `` |
| `scene` | `float` | Minimum changes between 2 frames to do a new OCR. The accepted range is 0-1 inclusive. Use 0 to force OCR on every frame | `0.02` |
| `lyrics_file` | `TextIO` | A file opened as texts to match the extracted texts. Requirements is same as ``--lyrics`` from cli. | `None` |
| `sub_from_prev` | `float` | Seconds to subtract from each cue. | `0` |
| `min_to_sub` | `float` |  The minimum length in seconds of each cue to apply `sub_from_prev`. | `0` |
| `crop` | `str` | The box to crop the video to. This value is passed to ffmpeg. The syntax is `width:height:x_pos:y_pos`. | `in_w:in_h:0:0` |
| `lang` | `list[str]` | A list of languages used by the OCR. | `None` |
| `write_file` | `bool` | Controls weather to write th result. | `True` |
| `out_format` | `str` | The format used to save the output. Accepts `ass`, `srt`, `vtt`, and `webvtt`. | `vtt` |


# Dependencies & Attribution**
This project uses the following third-party libraries:

- OCRMac (MIT) – https://github.com/straussmaximilian/ocrmac
- Pillow (MIT) – https://python-pillow.org
- tqdm (MPL 2.0, unmodified) – https://github.com/tqdm/tqdm
- NumPy (BSD) – https://numpy.org
- Python Standard Library (PSF License)


# License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
