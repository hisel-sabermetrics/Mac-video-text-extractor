"""
Date: 19 Jun 2025
"""

from argparse import ArgumentParser, FileType
from atexit import register
from collections.abc import Iterable
from difflib import get_close_matches
from itertools import count
from math import inf
from os import path, system
from re import escape, findall, search, split, sub
from subprocess import DEVNULL, PIPE, run
from typing import BinaryIO, Tuple

from ffmpeg import input
from numpy import append, array, frombuffer, object_, uint8, where
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
    frame_matrix: NDArray = cast_to_ndarray.reshape(
        (num_frame, height, width, 3)
    )

    return [
        fromarray(frame_matrix[i, start_y:end_y, start_x:end_x, :])
        for i in range(num_frame)
    ]


def extract_txt(
    start_sec: float,
    video_path_escaped: str,
    width: int,
    height: int,
    start_x: int,
    start_y: int,
    end_x: int,
    end_y: int,
    lang: list[str] | None = None,
) -> str:
    img: Image = video_to_frames(
        video_path_escaped,
        start_sec,
        height,
        width,
        1,
        start_x,
        start_y,
        end_x,
        end_y,
    )[0]

    txt_list: list[tuple(str, float, list(float))] = ocrmac.OCR(
        img, language_preference=lang
    ).recognize()

    if txt_list is False:
        return

    return "\n".join(txt[0] for txt in txt_list)


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
    try:
        open(
            video_path := path.expanduser(
                sub(r"^.+?\.", "ttv.", video_path[::-1])[::-1]
            ),
            "x",
        ).write(file_vtt)
    except FileExistsError:
        video_path: str = video_path[:-4]
        for i in count(start=2):
            try:
                open(
                    video_path := video_path + f"-{i}.vtt",
                    "x",
                ).write(file_vtt)
            finally:
                break
    finally:
        print("Saved to " + sub(r"\\(?!\\)", "", video_path))


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

    with tqdm(total=num_seg) as pbar:
        # screenshot_path = f"{dir_temp}/screenshot.png"
        first_cuenot__entered: bool = True
        for start, end in zip(time_start, time_end):
            this_cue = extract_txt(
                start,
                video_path_escaped,
                width,
                height,
                start_x,
                start_y,
                end_x,
                end_y,
                lang,
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
