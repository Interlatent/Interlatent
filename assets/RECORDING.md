# Recording the README capability GIFs

The four cards in the README's [About](../README.md#about) section each carry a commented-out
`<img>` tag. Record the clip, encode it to the filename below, drop it in this directory, and
uncomment the tag. Nothing else in the README changes.

## Budget

Committed GIFs live in git history permanently, so each one is capped:

| | Target |
|---|---|
| Width | 480 px (displays at 400 px, so there is no visible loss) |
| Frame rate | 10 fps |
| Length | ~4 s |
| File size | **1.5 MB max** |

That is ~6 MB total across the four. If a clip lands over budget, cut its length before you
cut its dimensions — a shorter clip reads better in a README than a smaller one.

## Encoding

Two-pass palette generation; a naive `ffmpeg -i in.mp4 out.gif` will be several times larger
and visibly dithered.

```bash
ffmpeg -i clip.mp4 -vf "fps=10,scale=480:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse" out.gif
```

Trim first if the source is long (`-ss 00:00:03 -t 4`). Check the result with `du -h out.gif`
before committing.

## Shot list

### `control.gif` — Control any arm
Two different arms (e.g. SO-101 and YAM) running the same script side by side, with only the
kind argument differing. Frame both arms in one shot if possible; a split screen otherwise.
This is the clip that proves the section's central claim, so it is worth the most effort.

### `behavior.gif` — Named moves
A single arm running a packaged behavior — the SO-101 `hello` wave is the obvious one. Frame
tight on the arm. Cheapest clip to shoot and it matches the Quickstart code exactly.

### `teleop.gif` — Teleoperate and collect
Someone in a headset driving an arm. Split screen (headset view beside the real robot) is far
stronger than either half alone, because the point is that the two move together. If a
takeover from a running policy is easy to stage, that is the better 4 seconds.

### `policy.gif` — Run a policy
An arm completing a task autonomously under a policy. Motion continuity is the message here,
so favour an uninterrupted 4 seconds of a single motion over a cut-together summary of a
longer task.

## Alt text

Each commented tag already carries `alt` text describing the shot. If you change what a clip
shows, update the `alt` to match — it is what screen readers and failed image loads surface.
