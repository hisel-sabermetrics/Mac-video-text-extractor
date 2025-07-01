"""
Date: 19 Jun 2025
"""

from argparse import ArgumentParser, FileType
from difflib import get_close_matches
from math import inf
from os import path, system
from re import escape, findall, search, split, sub
from subprocess import DEVNULL, PIPE, run
from typing import BinaryIO

from ffmpeg import input
from numpy import finfo, float32, frombuffer, uint8
from numpy.typing import NDArray
from ocrmac import ocrmac
from PIL import Image
from PIL.Image import fromarray
from tqdm import tqdm


def escape_path(raw_path: str) -> str:
    unescape_user_dir = sub(r"\\~", "~", escape(raw_path))
    escape_quote = sub('"', r"\"", unescape_user_dir)
    return sub(r"\\\.(?=\w+$)", ".", escape_quote)


def sec2str_time(sec: float) -> str:
    h: int = int(sec // 3600)
    min: int = int((sec % 3600) // 60)
    s: float = sec % 60
    return f"{h:02}:{min:02}:{s:06.3f}"


def extract_txt(
    start_sec: float, video_path_escaped: str, width: int, height: int
) -> str:
    out: str = run(
        f"ffmpeg -ss "
        + start_sec
        + " -i "
        + video_path_escaped
        + " -frames:v 1 -f rawvideo -pix_fmt rgb24 pipe:",
        stdout=PIPE,
        stderr=DEVNULL,
        shell=True,
    ).stdout

    cast_to_ndarray: NDArray = frombuffer(out, uint8)

    frame_matrix: NDArray = cast_to_ndarray.reshape([height, width, 3])

    img: Image = fromarray(frame_matrix)

    txt_list: list[tuple(str, float, list(float))] = ocrmac.OCR(
        img
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
        [r"(?<=\w)'((?=s\b)|\b)", "\u201d"],  # Right double quote
        [r"((?m:^)|((?<=\s)))'(?=\w)", "\u201c"],  # Left double quote
    ]
    for replacing in replacement_list:
        txt_joined: str = sub(replacing[0], replacing[1], txt_joined)
    return txt_joined


def seg_from_scene(video_path_escaped: str, scene: float) -> str:
    return run(
        "ffmpeg -i " + video_path_escaped + " -filter:v "
        '"hue=s=0,'  # Turn greyscale
        "maskfun=low=230:high=230:fill=0:sum=255,"  # Mask
        f"select='gt(scene,{scene})',showinfo\" -f null -",
        stdout=DEVNULL,
        stderr=PIPE,
        text=True,
        shell=True,
    ).stderr


def seg_from_key(video_path_escaped: str) -> str:
    return run(
        "ffmpeg -i " + video_path_escaped
        # + " -filter:v \"select='eq(key,1)',showinfo\" -f null -",
        + ' -filter:v "showinfo" -f null -',
        stdout=DEVNULL,
        stderr=PIPE,
        text=True,
        shell=True,
    ).stderr


def video2vtt(
    video_path: str,
    remove: str = "",
    replace: str = "",
    scene: float = 0.02,
    lyrics_file: BinaryIO | None = None,
) -> None:
    use_ltrics: bool = lyrics_file is not None
    if use_ltrics:
        lyrics_list: set[str] = set(split("\n{2,}", lyrics_file.read()))
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
            seg_change: str = seg_from_scene(video_path_escaped, scene)
        else:
            scene = 0
            seg_change: str = seg_from_key(video_path_escaped)

        time_start: list[str] = [
            t for t in findall(r"(?<=pts_time:)[0-9\.]++", seg_change)
        ]
        time_end: list[str] = time_start[1:]
        time_start.pop()
        num_seg: int = len(time_end)
        if scene != 0 and num_seg > int(video_meta["nb_read_frames"]) * 0.8:
            scene = 0
        else:
            break

    width: int = video_meta["width"]
    height: int = video_meta["height"]
    print(f"Number of scene found: {num_seg}")

    cue_list: list[list[float, float, str]] = []

    with tqdm(total=num_seg) as pbar:
        # screenshot_path = f"{dir_temp}/screenshot.png"
        this_cue: str = extract_txt(
            time_start[0], video_path_escaped, width, height
        )
        if remove != "":
            this_cue = sub(remove, replace, this_cue)
        if use_ltrics:
            this_cue = get_close_matches(this_cue, lyrics_list, 1, 0)[0]
        cue_list: list[list[float, float, str]] = [
            [
                float(time_start.pop(0)),
                float(time_end.pop(0)),
                # pretty(this_cue),
                this_cue,
            ],
        ]
        pbar.update()
        for start, end in zip(time_start, time_end):
            this_cue = extract_txt(start, video_path_escaped, width, height)
            if remove != "":
                this_cue = sub(remove, replace, this_cue)
            if this_cue == "":
                pbar.update()
                continue
            if use_ltrics:
                this_cue = get_close_matches(this_cue, lyrics_list, 1, 0)[0]
            if this_cue == cue_list[-1][2]:
                cue_list[-1][1] = float(end)
                pbar.update()
                continue
            if cue_list[-1][1] - cue_list[-1][0] > 0.15:
                cue_list[-1][1] -= 0.1
            # cue_list.append([float(start), float(end), pretty(this_cue)])
            cue_list.append([float(start), float(end), this_cue])
            pbar.update()

    file_vtt = "WEBVTT\n\n" + "\n\n".join(
        "{0} --> {1}\n{2}".format(
            sec2str_time(cue[0]), sec2str_time(cue[1]), cue[2]
        )
        for cue in cue_list
    )

    open(
        path.expanduser(sub(r"^.+?\.", "ttv.", video_path[::-1])[::-1]),
        "x",
    ).write(file_vtt)

    # system(f"rm -rf {dir_temp} >/dev/null 2>&1")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("path", nargs="+", type=str)
    parser.add_argument("--pattern", nargs="*", type=str, default=["", ""])
    parser.add_argument(
        "--lyrics", nargs="*", type=FileType("r"), default=None
    )
    parser.add_argument("--min-change", nargs="?", type=float, default=0.02)
    args = parser.parse_args()
    if len(args.pattern) == 1:
        args.pattern = [args.pattern[0], ""]

    lyrics_set: list[BinaryIO | None] = [None] * len(args.path)
    if args.lyrics is not None:
        for i in range(len(args.lyrics)):
            lyrics_set[i] = args.lyrics[i]
    for file_path, lyrics_path in zip(args.path, lyrics_set):
        video2vtt(
            file_path,
            args.pattern[0],
            args.pattern[1],
            args.min_change,
            lyrics_path,
        )
