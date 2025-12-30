"""
Date: 19 Jun 2025
"""

from argparse import ArgumentParser, FileType
from atexit import register
from collections.abc import Iterable, Iterator
from concurrent.futures import Future, ProcessPoolExecutor, as_completed, wait
from difflib import get_close_matches
from itertools import chain, count, repeat
from math import inf
from os import path, system
from os.path import exists, expanduser
from re import escape, findall, search, split, sub
from subprocess import DEVNULL, PIPE, run
from typing import Any, BinaryIO, Optional, Tuple

from ffmpeg import input
from numpy import (
    append,
    array,
    bool_,
    concatenate,
    diff,
    flatnonzero,
    float32,
    frombuffer,
    int8,
    intp,
    isin,
    object_,
    r_,
    uint8,
    where,
)
from numpy._typing import _32Bit
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
    path_escaped: str,
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
        f"ffmpeg -ss {start_sec} -i "
        + path_escaped
        + f" -frames:v {num_frame} -f rawvideo -pix_fmt rgb24 pipe:",
        stdout=PIPE,
        stderr=DEVNULL,
        shell=True,
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
    img: list[Image, ...], lang: list[str, ...] | None = None, workers: int = 3
) -> Iterator[str, ...]:
    task_len: int = len(img)
    return ProcessPoolExecutor(max_workers=min(task_len, workers)).map(
        text_from_1_img,
        img,
        repeat(lang),
        chunksize=max(1, task_len // (workers * 4)),
    )


def extract_txt(
    start_sec: float,
    video_path_escaped: str,
    width: int,
    height: int,
    num_frame: int = 1,
    frame_index: Iterable[int] = [0],  # index (from start_sec) of frames kept
    start_x: int = 0,
    start_y: int = 0,
    end_x: int | None = None,
    end_y: int | None = None,
    lang: list[str] | None = None,
) -> Iterator[str, ...]:
    return text_from_img(
        video_to_frames(
            video_path_escaped,
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


def seg_from_scene(video_path_escaped: str, scene: float, crop: str) -> str:
    return run(
        "ffmpeg -i " + video_path_escaped + " -filter:v "
        f'"crop={crop},'
        "hue=s=0,"  # Turn greyscale
        "maskfun=low=230:high=230:fill=0:sum=255,"  # Mask
        f"select='gt(scene,{scene})',showinfo\" -f null -",
        stdout=DEVNULL,
        stderr=PIPE,
        text=True,
        shell=True,
    ).stderr


def seg_from_key(video_path_escaped: str, crop) -> str:
    return run(
        "ffmpeg -i " + video_path_escaped
        # + " -filter:v \"select='eq(key,1)',showinfo\" -f null -",
        + ' -filter:v "' f"crop={crop},showinfo" '" -f null -',
        stdout=DEVNULL,
        stderr=PIPE,
        text=True,
        shell=True,
    ).stderr


def video_length(video_path_escaped: str) -> float:
    return float(
        run(
            "ffprobe -v error "
            "-select_streams v:0 "
            "-show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "
            + video_path_escaped,
            stdout=PIPE,
            stderr=DEVNULL,
            text=True,
            shell=True,
        ).stdout.strip("\n")
    )


def all_frame_timestame(video_path_escaped: str) -> NDArray[float32]:
    return array(
        findall(
            r"(?<=pts_time:)[0-9\.]++",
            run(
                "ffmpeg -i "
                + video_path_escaped
                + " -filter:v showinfo -f null -",
                stdout=DEVNULL,
                stderr=PIPE,
                text=True,
                shell=True,
            ).stderr,
        ),
        dtype=float32,
    )


def keyframe_timestamp(video_path_escaped: str) -> list[float]:
    return [
        float(t)
        for t in findall(
            r"(?<=pts_time:)[0-9\.]++",
            run(
                "ffmpeg -i "
                + video_path_escaped
                + " -filter:v \"select='eq(key,1)',showinfo\" -f null -",
                stdout=DEVNULL,
                stderr=PIPE,
                text=True,
                shell=True,
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


_ALL_FRAME_START_TIME: Optional[str] = None
_VIDEO_PATH_ESCAPED: Optional[str] = None
_WIDTH: Optional[int] = None
_HEIGHT: Optional[int] = None
_START_X: Optional[int] = None
_START_Y: Optional[int] = None
_END_X: Optional[int | None] = None
_END_Y: Optional[int | None] = None
_LANG: Optional[list[str] | None] = None


def _init_worker(
    all_frame_start_time: str,
    video_path_escaped: str,
    width: int,
    height: int,
    start_x: int,
    start_y: int,
    end_x: int | None,
    end_y: int | None,
    lang: list[str] | None,
) -> None:
    global _ALL_FRAME_START_TIME, _VIDEO_PATH_ESCAPED, _WIDTH, _HEIGHT
    global _START_X, _START_Y, _END_X, _END_Y, _LANG
    _ALL_FRAME_START_TIME = all_frame_start_time
    _VIDEO_PATH_ESCAPED = video_path_escaped
    _WIDTH = width
    _HEIGHT = height
    _START_X = start_x
    _START_Y = start_y
    _END_X = end_x
    _END_Y = end_y
    _LANG = lang


def _timestamps_to_text(
    timestamps: NDArray[float32],
) -> tuple[str]:
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
    texts: Iterator[str] = extract_txt(
        timestamps[0],
        _VIDEO_PATH_ESCAPED,
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
    return tuple(texts)


def write_to_file(
    video_path: str, cue_list: Iterable[Iterable[float, float, str]]
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
    if not exists(path_before_i + ".vtt"):
        open(
            path_before_i + ".vtt",
            "x",
        ).write(file_vtt)
        return
    for i in count(2):
        if exists(path_before_i + f"-{i}.vtt"):
            continue
        open(path_before_i + f"-{i}.vtt", "x").write(file_vtt)
        return


def video2vtt(
    video_path: str,
    remove: str = "",
    replace: str = "",
    scene: float = 0.02,
    lyrics_file: BinaryIO | None = None,
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
    # root: str = sub(r"^.+?\/(?!\/)", "", path.realpath(__file__)[::-1])[::-1]
    # dir_temp = f"{root}/temp"
    video_path_escaped = escape_path(video_path)
    # system(f"mkdir -p '{dir_temp}'")

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

    num_seg = inf
    while True:
        if scene != 0:
            seg_change: str = seg_from_scene(video_path_escaped, scene, crop)
        else:
            scene = 0
            seg_change: str = seg_from_key(video_path_escaped, crop)

        time_start: list[float, ...] = [
            float(t) for t in findall(r"(?<=pts_time:)[0-9\.]++", seg_change)
        ]
        time_end: list[float, ...] = time_start[1:]
        time_start.pop()
        num_seg: int = len(time_end)
        if scene != 0 and num_seg > int(video_meta["nb_read_frames"]) * 0.8:
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

    if use_ltrics:
        with tqdm(total=num_seg) as pbar:
            # screenshot_path = f"{dir_temp}/screenshot.png"
            first_cuenot__entered: bool = True
            for start, end in zip(time_start, time_end):
                this_cue: str = next(
                    extract_txt(
                        start,
                        video_path_escaped,
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
    else:
        keyframe_start_time: list[float, ...] = keyframe_timestamp(
            video_path_escaped
        ) + [video_length(video_path_escaped) + 1]
        if keyframe_start_time[0] != 0:
            keyframe_start_time = [0] + keyframe_start_time

        all_frame_start_time: NDArray[float32] = all_frame_timestame(
            video_path_escaped
        )
        # Cast to NDArray
        time_start: NDArray = array(time_start, float32)
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
                video_path_escaped,
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
                video_path_escaped,
                width,
                height,
                start_x,
                start_y,
                end_x,
                end_y,
                lang,
            )
            with tqdm(total=num_seg) as pbar:
                workers: list[Future[tuple[NDArray[float32], tuple[str]]]] = [
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

        # Remove all empty cue
        print("Removing empty cue")
        mask: NDArrat[bool_] = all_cue != ""
        time_start = time_start[mask]
        time_end = array(time_end, dtype=float32)[mask]
        all_cue = all_cue[mask]
        del mask

        # Merge based on ocr
        print("Merging cue")
        time_start, time_end, all_cue = merge_cue(
            time_start, time_end, all_cue
        )

        # Merge connected and identical
        cue_list = tuple(zip(time_start, time_end, all_cue))

    # Subtract from cue if it is set
    if sub_from_prev != 0:
        print("Adjusting end time")
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
        print("Writing to file")
        write_to_file(video_path, cue_list)
        _WRITE_TO_FILE = False  # Suppress writing another copy

    # system(f"rm -rf {dir_temp} >/dev/null 2>&1")


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
    args = parser.parse_args()
    if len(args.pattern) == 1:
        args.pattern = [args.pattern[0], ""]
    if args.lang == ["None"]:
        args.lang = None

    lyrics_set: list[BinaryIO | None] = [None] * len(args.path)
    if args.lyrics is not None:
        for i in range(len(args.lyrics)):
            lyrics_set[i] = args.lyrics[i]
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
        )
