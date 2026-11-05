# The X-Ray Effect

This folder puts the x-ray flicker on the user's own talking head clip. When the user gives you a clip or asks for the
effect, do the steps below. Keep it simple for them: no explaining the code, no extra choices beyond the ones here.

## What they can choose

Only ask if they haven't said, and only these. If they just say "do it", use the defaults.

WHEN: [the start of the clip]
→ "the start of the clip", or seconds in, like 12. Goes in as --start.

COLOUR: [#1C3A7C]
→ The colour behind every second flash. A hex code goes in exactly; turn a colour name into a hex code and tell them which. Goes in as --colour.

## The steps

1. Get the clip. They drag it into the chat, which gives you its path. If there is no clip, ask for one: one person, chest
   up, facing the camera, hands visible. Never change or move their original file.

2. Run the one command, from this folder, quoting the path:

   python3 run.py "/path/to/their clip.mp4" --start 0 --colour "#1C3A7C"

   On Windows use `python` if `python3` is missing. The first run sets itself up inside this folder and takes a few
   minutes, so run it in the background and tell them it's the first-time setup. After that, a short clip takes about a minute. Warning lines from the tracking library are normal; only an X-RAY STOPPED line or an error exit means something went wrong. It needs no admin rights or password.
   If there is no Python at all, install uv from https://docs.astral.sh/uv/getting-started/installation/ (no admin rights),
   then run: uv run --no-project --python 3.12 run.py "/path/to/their clip.mp4"

3. Look before you hand it over. Open `renders/<clip name>-xray-check.jpg`, which shows every look, and check:
   - the skeleton sits inside the body: skull on the face, ribcage in the chest, arm bones in the arms
   - only the person changes; the room, the table and everything else stays the normal video
   - no large part of the person is missing, and no part of the room is treated as the person

   If it fails, try a different --start, where the person is facing the camera with their hands more still. Fast hands
   can lose their finger bones, which is normal for a flash this short and isn't worth stopping for. Only try twice, then give them
   the best one and say what's off.

4. Give them the file: `renders/<clip name>-xray.mp4`. It has the whole clip, the same frame rate and their audio. Open the
   renders folder for them. Tell them to drop it straight into their editing timeline. Don't send the check picture unless they ask.

## Tweaks they might ask for

- "Start it at 12 seconds": --start 12
- "No zoom": --punch 1
- "Less zoom": --punch 1.25
- "Change the blue": --colour with the new hex
- Anything else about the look lives in THE LOOK at the top of `xray.py` and the look functions below it. Change as little as you can.

## How it should look

Ten flashes in 0.8 seconds, on every second frame of a 24 fps grid, starting two frames after --start: x-ray, cyan
negative, line drawing, x-ray on colour, cyan negative, line drawing on colour, x-ray, cyan negative, line drawing, x-ray
on colour. The camera accelerates from the whole frame into the nose bridge and lands right after the last flash (up to
1.5x, less if the head would leave the frame), creeps 3 % for 0.95 s, then eases back over 0.9 s.

## What it needs

Python, plus MediaPipe, OpenCV and a copy of ffmpeg, all free, all installed inside this folder. Nothing is uploaded and no
account or API key is needed. It runs on Apple Silicon Macs, Windows and Linux. MediaPipe has no build for Intel Macs, so tell
those users it won't run there.
