# Put your video clips here

This folder ships empty. Copy your footage into it:

    firebrand-kit/clips/eaton_01.mp4
    firebrand-kit/clips/eaton_02.mp4

Then everything else in the kit refers to them as `clips/eaton_01.mp4`.

You do not have to use this folder — every command takes any path you give it,
including one somewhere else on your disk or a `gs://` URL. It exists so that
`bash cloud/upload.sh clips/` has an obvious thing to upload, and so the
examples in the README are literally runnable.

Windows / Git Bash note: quote paths containing spaces, and `C:` is `/c/`:

    python run_pipeline.py "/c/Users/Your Name/Downloads/cam1.mp4" --out work/x

## Formats

Whatever OpenCV can decode: .mp4, .avi, .mov, .mkv. Some DVRs export .dav or
H.265, which it usually cannot read. Convert first, and keep the quality high —
compression is what destroys objects this small:

    ffmpeg -i input.dav -c:v libx264 -crf 18 -an clips/cam1.mp4

Use the ORIGINAL files, not a re-export at the DVR software's default quality.
