"""
My Python Script (MIT License)
Copyright (c) 2025 Your Name

This script uses the following third-party libraries:

- OCRMac (MIT) https://github.com/straussmaximilian/ocrmac
- Pillow (MIT) https://python-pillow.org
- tqdm (MPL 2.0, unmodified) https://github.com/tqdm/tqdm
- NumPy (BSD) https://numpy.org
- Python Standard Library (PSF License)

This script also requires the following CLI tool:

- ffmpeg https://ffmpeg.org/ (for video/audio processing)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files, to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of the Software.
"""

from argparse import ArgumentParser, FileType
from atexit import register
from collections.abc import Iterable
from concurrent.futures import Future, ProcessPoolExecutor, as_completed, wait
from difflib import get_close_matches
from itertools import count, repeat
from os.path import exists, expanduser
from re import escape, findall, split, sub
from subprocess import DEVNULL, PIPE, run
from typing import Any, TextIO, Tuple

from numpy import (
    append,
    array,
    bool_,
    concatenate,
    diff,
    flatnonzero,
    float32,
    frombuffer,
    fromiter,
    int8,
    intp,
    isin,
    object_,
    r_,
    uint8,
    where,
)
from numpy.typing import NDArray
from ocrmac import ocrmac
from PIL import Image
from PIL.Image import fromarray
from tqdm import tqdm

_WRITE_TO_FILE: bool = True  # Flag to allow write


def escape_path(raw_path: str) -> str:
    unescape_user_dir = sub(r"\\~", "~", escape(raw_path))
    escape_quote = sub('"', r"\"", unescape_user_dir)
    return sub(r"\\\.(?=\w+$)", ".", escape_quote)


def sec2str_time(sec: float) -> str:
    h: int = int(sec // 3600)
    min: int = int((sec % 3600) // 60)
    s: float = sec % 60
    return f"{h:02}:{min:02}:{s:06.3f}"


def video_to_frames(
    path: str,
    start_sec: float,
    height: int,
    width: int,
    num_frame: int = 1,
    frame_index: Iterable[int] = [0],  # index (from start_sec) of frames kept
    start_x: int = 0,
    start_y: int = 0,
    end_x: int | None = None,
    end_y: int | None = None,
) -> list[Image]:
    buffer: str = run(
        [
            "ffmpeg",
            "-ss",
            str(start_sec),
            "-i",
            path,
            "-frames:v",
            str(num_frame),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:",
        ],
        stdout=PIPE,
        stderr=DEVNULL,
    ).stdout

    cast_to_ndarray: NDArray = frombuffer(buffer, uint8)
    del buffer

    frame_matrix: NDArray = cast_to_ndarray.reshape(
        (num_frame, height, width, 3)
    )

    return [
        fromarray(frame_matrix[i, start_y:end_y, start_x:end_x, :])
        for i in frame_index
    ]


def text_from_1_img(img: Image, lang: list[str, ...] | None = None) -> str:
    return "\n".join(
        ocrmac.OCR(
            img,
            language_preference=lang,
            detail=False,
        ).recognize()
    )


def text_from_img(
    imgs: list[Image, ...],
    lang: list[str, ...] | None = None,
    workers: int = 3,
) -> list[str, ...]:
    task_len: int = len(imgs)
    pool = ProcessPoolExecutor(max_workers=min(task_len, workers))
    results: list[Future[str]] = [
        pool.submit(text_from_1_img, img, lang) for img in imgs
    ]
    tasks_to_check: list[Future[str]] = results
    timeout: list[Future[str]] = []
    i_to_check: list[int] = list(range(task_len))
    i_timeout: list[int] = []
    texts: list[str | None] = [None] * task_len
    while tasks_to_check:
        for result in as_completed(tasks_to_check, 5):
            i: int = i_to_check[tasks_to_check.index(result)]
            # No timeout
            if result.exception() is None:
                texts[i] = result.result()
                continue
            # Restart task and track
            timeout.append(pool.submit(text_from_1_img, imgs[i], lang))
            i_timeout.append(i)
        # Only check timed out tasks
        tasks_to_check = timeout
        i_to_check = i_timeout

    return texts


def extract_txt(
    start_sec: float,
    video_path: str,
    width: int,
    height: int,
    num_frame: int = 1,
    frame_index: Iterable[int] = [0],  # index (from start_sec) of frames kept
    start_x: int = 0,
    start_y: int = 0,
    end_x: int | None = None,
    end_y: int | None = None,
    lang: list[str] | None = None,
) -> list[str, ...]:
    return text_from_img(
        video_to_frames(
            video_path,
            start_sec,
            height,
            width,
            num_frame,
            frame_index,
            start_x,
            start_y,
            end_x,
            end_y,
        ),
        lang,
    )


def pretty(txt_joined: str) -> str:
    replacement_list: list[list[str, str]] = [
        [r"\-", "—"],  # unescape dash
        [r"(\s+-\s+)|(?m:\s*-$)", "\u2014"],  # Dash to em dash
        [r"(?<=\w)'((?=s\b)|\b)", "\u2019"],  # Right apostrophe
        [r"((?m:^)|((?<=\s)))'(?=\w)", "\u2018"],  # Left apostrophe
        [r'(?<=\w)"\b', "\u201d"],  # Right double quote
        [r'\b"(?=\w)', "\u201c"],  # Left double quote
    ]
    for replacing in replacement_list:
        txt_joined: str = sub(replacing[0], replacing[1], txt_joined)
    return txt_joined


def seg_from_scene(
    video_path: str, scene: float, crop: str
) -> NDArray[float32]:
    return array(
        [
            float(t)
            for t in findall(
                r"(?<=pts_time:)[0-9\.]++",
                run(
                    [
                        "ffmpeg",
                        "-i",
                        video_path,
                        "-filter:v",
                        f"crop={crop},"
                        "hue=s=0,"  # Turn greyscale
                        "maskfun=low=230:high=230:fill=0:sum=255,"  # Mask
                        f"select='gt(scene,{scene})',showinfo",
                        "-f",
                        "null",
                        "-",
                    ],
                    stdout=DEVNULL,
                    stderr=PIPE,
                    text=True,
                ).stderr,
            )
        ],
        float32,
    )


def video_length(video_path: str) -> float:
    return float(
        run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            stdout=PIPE,
            stderr=DEVNULL,
            text=True,
        ).stdout.strip("\n")
    )


def all_frame_timestamp(video_path: str) -> NDArray[float32]:
    return array(
        findall(
            r"(?<=pts_time:)[0-9\.]++",
            run(
                [
                    "ffmpeg",
                    "-i",
                    video_path,
                    "-filter:v",
                    "showinfo",
                    "-f",
                    "null",
                    "-",
                ],
                stdout=DEVNULL,
                stderr=PIPE,
                text=True,
            ).stderr,
        ),
        dtype=float32,
    )


def keyframe_timestamp(video_path: str) -> list[float]:
    return [
        float(t)
        for t in findall(
            r"(?<=pts_time:)[0-9\.]++",
            run(
                [
                    "ffmpeg",
                    "-i",
                    video_path,
                    "-filter:v",
                    "select='eq(key,1)',showinfo",
                    "-f",
                    "null",
                    "-",
                ],
                stdout=DEVNULL,
                stderr=PIPE,
                text=True,
            ).stderr,
        )
    ]


# Get values within boundary
def extract_between_value(
    value_list: NDArray[float32], start_inclusive: float, end_exclusive: float
) -> NDArray[float32]:
    return value_list[
        (start_inclusive <= value_list) & (value_list < end_exclusive)
    ]


def merge_cue(
    time_start: NDArray[float32],
    time_end: NDArray[float32],
    all_cue: NDArray[str],
) -> tuple[NDArray[float32], NDArray[float32], NDArray[str]]:

    # These 3 masks are 1 less in len along axis 0
    start_end_conected: NDArray[bool_] = time_start[1:] == time_end[:-1]
    cue_adj_identical: NDArray[bool_] = all_cue[1:] == all_cue[:-1]
    connected_and_identical: NDArray[bool_] = (
        start_end_conected & cue_adj_identical
    )
    del start_end_conected, cue_adj_identical

    keep: NDArray[bool] = r_[True, ~connected_and_identical]
    # All cues unique
    if all(keep):
        return time_start, time_end, all_cue

    # Set end time to the end of connected and identical
    diff_in_state: NDArray[int8] = diff(
        connected_and_identical.astype(int8), prepend=0, append=0
    )
    del connected_and_identical
    i_before_con: NDArray[intp] = flatnonzero(diff_in_state == 1)
    i_end_con: NDArray[intp] = flatnonzero(diff_in_state == -1)
    # Extend cue end time
    time_end[i_before_con] = time_end[i_end_con]
    del diff_in_state, i_before_con, i_end_con

    # Remove all connected and identical cue after the first cue
    return time_start[keep], time_end[keep], all_cue[keep]


_ALL_FRAME_START_TIME: str | None = None
_VIDEO_PATH: str | None = None
_WIDTH: int | None = None
_HEIGHT: int | None = None
_START_X: int | None = None
_START_Y: int | None = None
_END_X: int | None = None
_END_Y: int | None = None
_LANG: list[str] | None = None


def _init_worker(
    all_frame_start_time: str,
    video_path: str,
    width: int,
    height: int,
    start_x: int,
    start_y: int,
    end_x: int | None,
    end_y: int | None,
    lang: list[str] | None,
) -> None:
    global _ALL_FRAME_START_TIME, _VIDEO_PATH, _WIDTH, _HEIGHT
    global _START_X, _START_Y, _END_X, _END_Y, _LANG
    _ALL_FRAME_START_TIME = all_frame_start_time
    _VIDEO_PATH = video_path
    _WIDTH = width
    _HEIGHT = height
    _START_X = start_x
    _START_Y = start_y
    _END_X = end_x
    _END_Y = end_y
    _LANG = lang


def _timestamps_to_text(
    timestamps: NDArray[float32],
) -> list[str]:
    # Get index of frames
    index: NDArray[intp] = flatnonzero(
        isin(
            _ALL_FRAME_START_TIME,
            timestamps,
            True,
            kind="sort",
        )
    )
    # Reset 0 to first frame
    index = index - index[0]
    # Get num frames from first frame to last inclusive
    num_frames: int = index[-1] + 1
    # Get text
    return extract_txt(
        timestamps[0],
        _VIDEO_PATH,
        _WIDTH,
        _HEIGHT,
        num_frames,
        index,
        _START_X,
        _START_Y,
        _END_X,
        _END_Y,
        _LANG,
    )


def write_to_file(
    video_path: str,
    cue_list: Iterable[Iterable[float, float, str]],
    format: str = "vtt",
) -> None:
    # Check if write is allowed
    if not _WRITE_TO_FILE:
        return
    file_vtt: str = "WEBVTT\n\n" + "\n\n".join(
        "{0} --> {1}\n{2}".format(
            sec2str_time(cue[0]), sec2str_time(cue[1]), cue[2]
        )
        for cue in cue_list
    )
    path_before_i: str = sub(r"\.[^\.]+$", "", expanduser(video_path))
    path_used: str = path_before_i + "." + format
    # Check if exists
    if exists(path_used):
        for i in count(2):
            path_used = path_before_i + f"-{i}." + format
            if exists(path_used):
                continue
            break

    if format != "vtt" and format != "webvtt":
        temp_path: str = path_used + "-1.vtt"
        if exists(temp_path):
            for i in count(2):
                temp_path = path_before_i + f"-{i}.vtt"
                if exists(temp_path):
                    continue
                break
        open(temp_path, "x").write(file_vtt)
        # Convert format
        run(
            ["ffmpeg", "-i", temp_path, path_used],
            stdout=DEVNULL,
            stderr=DEVNULL,
        ).stdout
        run(["rm", path_used], stdout=DEVNULL, stderr=DEVNULL).stdout
        return
    # No conversion
    open(path_used, "x").write(file_vtt)


def old_video2vtt(
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
) -> None:
    use_ltrics: bool = lyrics_file is not None
    if use_ltrics:
        lyrics_list: list[str] = [
            cue.strip() for cue in split("\n{2,}", lyrics_file.read())
        ]
        lyrics_file.close()

    video_meta: dict = eval(
        run(
            [
                "ffprobe",
                "-i",
                video_path,
                "-print_format",
                "json",
                "-loglevel",
                "fatal",
                "-show_streams",
                "-count_frames",
                "-select_streams",
                "v",
            ],
            stdout=PIPE,
            stderr=DEVNULL,
        ).stdout
    )["streams"][0]

    while True:
        time_start: list[float, ...] = (
            seg_from_scene(video_path, scene, crop).tolist()
            if scene != 0
            else all_frame_timestamp(video_path).tolist()
        )
        time_end: list[float, ...] = time_start[1:]
        time_start.pop()
        num_frame: int = len(time_end)
        if scene != 0 and num_frame > int(video_meta["nb_read_frames"]) * 0.8:
            scene = 0
        else:
            break

    width: int = video_meta["width"]
    height: int = video_meta["height"]
    crop_param: list[int] = [
        int(
            eval(
                sub(r"^out_[hw]=", "", param),
                {
                    "in_h": height,
                    "in_w": width,
                },
            )
        )
        for param in split(":", crop)
    ]
    start_x: int = 0 if len(crop_param) <= 2 else int(crop_param[2])
    start_y: int = 0 if len(crop_param) <= 3 else int(crop_param[3])
    end_x: int = int(crop_param[0]) + start_x
    end_y: int = (
        int(crop_param[0]) + start_y
        if len(crop_param) == 1
        else int(crop_param[1]) + start_y
    )

    print(f"Number of scene found: {num_seg}")

    with tqdm(total=num_seg) as pbar:
        # screenshot_path = f"{dir_temp}/screenshot.png"
        first_cuenot__entered: bool = True
        for start, end in zip(time_start, time_end):
            this_cue: str = next(
                extract_txt(
                    start,
                    video_path,
                    width,
                    height,
                    start_x=start_x,
                    start_y=start_y,
                    end_x=end_x,
                    end_y=end_y,
                    lang=lang,
                )
            )
            if remove != "":
                this_cue = sub(remove, replace, this_cue)
            if this_cue == "":
                pbar.update()
                continue
            if first_cuenot__entered:
                cue_list: list[float, float, str] = [
                    [
                        start,
                        end,
                        lyrics_list[0] if use_ltrics else this_cue,
                    ],
                ]
                if write_file:
                    # Write file at interrupt
                    # Does not work with sub_from_prev
                    _WRITE_TO_FILE = True
                    register(write_to_file, video_path, cue_list)
                pbar.update()
                first_cuenot__entered = False
                continue
            if use_ltrics:
                cue_to_check: set[str, str] = lyrics_list[
                    0 : len(cue_list) + 1
                ]
                this_cue = get_close_matches(this_cue, cue_to_check, 1, 0)[0]
            if this_cue == cue_list[-1][2]:
                cue_list[-1][1] = end
                pbar.update()
                continue
            # cue_list.append([float(start), float(end), pretty(this_cue)])
            cue_list.append([start, end, this_cue])
            pbar.update()

    # Subtract from cue if it is set
    if sub_from_prev != 0:
        # Cast to ndarray
        cue_list: NDArray[Tuple[int, 3], object_] = array(cue_list)
        start_time: NDArray[Tuple[int], float] = cue_list[:, 0].astype(float)
        end_time: NDArray[Tuple[int], float] = cue_list[:, 1].astype(float)
        end_time = where(
            (end_time == append(start_time[1:], 0))
            & (end_time - start_time >= min_to_sub),
            end_time - sub_from_prev,
            end_time,
        )
        cue_list = tuple(zip(start_time, end_time, cue_list[:, 2]))

    # Write to file
    if write_file:
        write_to_file(video_path, cue_list)
        _WRITE_TO_FILE = False  # Suppress writing another copy

    # system(f"rm -rf {dir_temp} >/dev/null 2>&1")


_LYRICS_LIST: list[str] = None
_SIZE_QUEUE: int = None


def _init_match(lyrics_list: list[str], size_queue: int) -> None:
    global _LYRICS_LIST, _SIZE_QUEUE
    _LYRICS_LIST = lyrics_list
    _SIZE_QUEUE = size_queue


def _get_match(
    cue: str,
    highest_i_used: int,
) -> tuple[str, int]:
    lyrics_checking: list[str] = _LYRICS_LIST[
        : highest_i_used + _SIZE_QUEUE * 2 + 3
    ]
    cue_obtained: list[str] = get_close_matches(
        cue,
        lyrics_checking,
        1,
        0.3,
    )
    if cue_obtained:
        cue_obtained = cue_obtained[0]
        return cue_obtained, lyrics_checking.index(cue_obtained)
    return "", 0


def remove_empty_line(string: str):
    # Strip whitespace at start and end
    string = string.strip()
    # Remove duplicate newline
    string = sub("\n+", "\n", string)
    # Squash all whitespace to 1 space
    string = sub("[\t ]+", " ", string)
    # Remove lines of only 1 space
    return string.replace("\n \n", "")


def video2vtt(
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
) -> None:
    """Extract texts from a video file and write to disk

    Keyword arguments:
    video_path    -- path to the video file
    remove        -- pattern to remove text
                     (default "")
    replace       -- the str literal to replace with
                     (default "")
    scene         -- the min diff between 2 fames before doing a new OCR
                     (default 0.02)
    lyrics_file   -- opened text file containing the texts to compare
                     (default None)
    sub_from_prev -- seconds to subtract before an immediate cue
                     (default 0)
    min_to_sub    -- min length in seconds to apply sub_from_prev
                     (default 0)
    crop          -- ffmpeg argument to crop the video
                     (default "in_w:in_h:0:0")
    lang          -- list of lang for OCR to use
                     (default None)
    write_file    -- bool flag for weather to write result
                     (default True)
    out_format    -- format to use for result
                     (default "vtt")
    """

    video_meta: dict = eval(
        run(
            [
                "ffprobe",
                "-i",
                video_path,
                "-print_format",
                "json",
                "-loglevel",
                "fatal",
                "-show_streams",
                "-count_frames",
                "-select_streams",
                "v",
            ],
            stdout=PIPE,
            stderr=DEVNULL,
        ).stdout
    )["streams"][0]

    time_start: NDArray[float32] = (
        seg_from_scene(video_path, scene, crop)
        if scene != 0
        else all_frame_timestamp(video_path)
    )
    time_end: NDArray[float32] = time_start[1:]
    time_start = time_start[:-1]
    num_frame: int = time_end.size

    width: int = video_meta["width"]
    height: int = video_meta["height"]
    crop_param: list[int] = [
        int(
            eval(
                sub(r"^out_[hw]=", "", param),
                {
                    "in_h": height,
                    "in_w": width,
                },
            )
        )
        for param in split(":", crop)
    ]
    start_x: int = 0 if len(crop_param) <= 2 else int(crop_param[2])
    start_y: int = 0 if len(crop_param) <= 3 else int(crop_param[3])
    end_x: int = int(crop_param[0]) + start_x
    end_y: int = (
        int(crop_param[0]) + start_y
        if len(crop_param) == 1
        else int(crop_param[1]) + start_y
    )

    print(f"Number of scene found: {num_frame}")

    keyframe_start_time: list[float, ...] = keyframe_timestamp(video_path) + [
        video_length(video_path) + 1
    ]
    if keyframe_start_time[0] != 0:
        keyframe_start_time = [0] + keyframe_start_time

    all_frame_start_time: NDArray[float32] = all_frame_timestamp(video_path)
    # Split time_start by keyframe
    timestamp_list: list[NDArray[float32]] = [
        time
        for i in range(1, len(keyframe_start_time))
        if (
            time := extract_between_value(
                time_start,
                keyframe_start_time[i - 1],
                keyframe_start_time[i],
            )
        ).size
        != 0  # Do nothing if this seg is not used
    ]
    del keyframe_start_time
    # Get time of each frame to extract text from in this seg
    print("Extracting texts")
    with ProcessPoolExecutor(
        3,
        initializer=_init_worker,
        initargs=(
            all_frame_start_time,
            video_path,
            width,
            height,
            start_x,
            start_y,
            end_x,
            end_y,
            lang,
        ),
    ) as pool:
        del (
            all_frame_start_time,
            width,
            height,
            start_x,
            start_y,
            end_x,
            end_y,
            lang,
        )
        with tqdm(total=num_frame) as pbar:
            workers: list[Future[tuple[str]]] = [
                pool.submit(_timestamps_to_text, timestamps)
                for timestamps in timestamp_list
            ]
            del timestamp_list
            # Update pbar
            for task in as_completed(workers):
                pbar.update(len(task.result()))

        # Join all output
        print("Collecting")
        all_cue: NDArray[str] = concatenate(
            [result.result() for result in workers],
            dtype=object,
            casting="safe",
        )

    # Apply replace
    if remove != "":
        if replace == "":
            print(f"Removing {remove}")
        else:
            print(f"Replacing {remove} with {replace}")
        # Merge first reduce tasks
        time_start, time_end, all_cue = merge_cue(
            time_start, time_end, all_cue
        )
        workers: int = 10
        all_cue = fromiter(
            ProcessPoolExecutor(workers).map(
                sub, repeat(remove), repeat(replace), all_cue
            ),
            object,
        )

        # Remove empty lines
        all_cue = fromiter(
            ProcessPoolExecutor(workers).map(remove_empty_line, all_cue),
            object,
        )

    # Remove all empty cue
    print("Removing empty cue")
    all_cue, time_start, time_end = remove_empty(all_cue, time_start, time_end)

    # Apply lyric file
    if lyrics_file is not None:
        lyrics_path: str = lyrics_file.name
        lyrics_list: list[str] = [
            cue.strip() for cue in split("\n{2,}", lyrics_file.read())
        ]
        lyrics_file.close()
        print(f"Applying lyrics from " + lyrics_path)

        # Merge first reduce tasks
        time_start, time_end, all_cue = merge_cue(
            time_start, time_end, all_cue
        )
        workers: int = 10
        len_queue: int = 3
        len_pending: int = workers + len_queue
        len_cue: int = all_cue.size

        with (
            ProcessPoolExecutor(
                max_workers=workers,
                initializer=_init_match,
                initargs=(lyrics_list, len_pending),
            ) as pool,
            tqdm(total=len_cue) as pbar,
        ):

            # Submit first batch
            cue_pending: list[Future[tuple[str, int]]] = [
                pool.submit(_get_match, all_cue[i], 0)
                for i in range(min(len_pending, len_cue))
            ]

            # Check length
            if len_cue <= len_pending:
                all_cue = cue_pending
                for _ in as_completed(cue_pending):
                    pbar.update()
            else:
                # Use producer consumer pattern if longer
                i: int = len_pending  # Track index
                highest_i_used: int = 0
                # Replace original value with Future
                all_cue[:len_pending]: NDArray[
                    str | Future[str, int]
                ] = cue_pending
                while cue_pending:
                    done: set[Future[tuple[str, int]]] = wait(
                        cue_pending, return_when="FIRST_COMPLETED"
                    )[0]
                    highest_i_used: int = max(
                        highest_i_used,
                        *[task.result()[1] for task in done],
                    )
                    # Producer
                    for task in done:
                        pbar.update()
                        cue_pending.remove(task)
                        # Submit new task
                        if i < len_cue:
                            all_cue[i] = pool.submit(
                                _get_match,
                                all_cue[i],
                                highest_i_used,
                            )
                            cue_pending.append(all_cue[i])
                            i = i + 1

        # Assign back to all cue
        all_cue: NDArray[str] = array(
            [cue_matched.result()[0] for cue_matched in all_cue]
        )

        # Remove empty cue
        all_cue, time_start, time_end = remove_empty(
            all_cue, time_start, time_end
        )

    # Merge connected and identical
    print("Merging cue")
    time_start, time_end, all_cue = merge_cue(time_start, time_end, all_cue)

    # Subtract from cue if it is set
    if sub_from_prev != 0:
        print("Adjusting end time")
        # Cast to ndarray
        time_end = where(
            (time_end == append(time_start[1:], 0))
            & (time_end - time_start >= min_to_sub),
            time_end - sub_from_prev,
            time_end,
        )
    cue_list: Tuple[Tuple[float, float, str]] = tuple(
        zip(time_start, time_end, all_cue)
    )

    # Write to file
    if write_file:
        print("Writing to file")
        write_to_file(video_path, cue_list, out_format)
        _WRITE_TO_FILE = False


def remove_empty(
    all_cue: NDArray[str], *all_array_to_apply: NDArray[Any]
) -> tuple[NDArray[str], NDArray[Any], ...]:
    mask: NDArray[bool_] = all_cue != ""
    yield all_cue[mask]
    for arr in all_array_to_apply:
        yield arr[mask]


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("path", nargs="+", type=str)
    parser.add_argument("--pattern", nargs="*", type=str, default=["", ""])
    parser.add_argument(
        "--lyrics", nargs="*", type=FileType("r"), default=None
    )
    parser.add_argument("--min-change", nargs="?", type=float, default=0.02)
    parser.add_argument("--crop", nargs="?", type=str, default="in_w:in_h:0:0")
    parser.add_argument("--lang", nargs="*", type=str, default=[None])
    parser.add_argument(
        "--subtract-from-pre", nargs="?", type=float, default=0
    )
    parser.add_argument(
        "--min-seg-len-for-subtract", nargs="?", type=float, default=0
    )
    parser.add_argument(
        "--version",
        nargs=1,
        type=int,
        choices=(1, 2),
        required=False,
        default=[2],
    )
    parser.add_argument(
        "--format",
        nargs=1,
        type=str,
        choices=("ass", "srt", "vtt", "webvtt"),
        required=False,
        default="vtt",
    )
    args = parser.parse_args()
    if len(args.pattern) == 1:
        args.pattern = [args.pattern[0], ""]
    if args.lang == ["None"]:
        args.lang = None

    lyrics_set: list[TextIO | None] = [None] * len(args.path)
    if args.lyrics is not None:
        for i in range(len(args.lyrics)):
            lyrics_set[i] = args.lyrics[i]
    match args.version[0]:
        case 1:
            for file_path, lyrics_path in zip(args.path, lyrics_set):
                old_video2vtt(
                    video_path=file_path,
                    remove=args.pattern[0],
                    replace=args.pattern[1],
                    scene=args.min_change,
                    lyrics_file=lyrics_path,
                    sub_from_prev=args.subtract_from_pre,
                    min_to_sub=args.min_seg_len_for_subtract,
                    crop=args.crop,
                    lang=args.lang,
                )
        case 2:
            for file_path, lyrics_path in zip(args.path, lyrics_set):
                video2vtt(
                    video_path=file_path,
                    remove=args.pattern[0],
                    replace=args.pattern[1],
                    scene=args.min_change,
                    lyrics_file=lyrics_path,
                    sub_from_prev=args.subtract_from_pre,
                    min_to_sub=args.min_seg_len_for_subtract,
                    crop=args.crop,
                    lang=args.lang,
                    out_format=args.format,
                )
