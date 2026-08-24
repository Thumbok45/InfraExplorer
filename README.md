# InfraExplorer

Local Windows program for long piezo / contact-mic recordings (FLAC, WAV, RF64, AIFF).

The original file is never overwritten.

## Start

1. Install [Python 3](https://www.python.org/downloads/) if you do not have it.
   During Windows setup, check **Add python.exe to PATH** and keep **tcl/tk**.
2. Download this repository as a zip (green **Code** button → **Download ZIP**).
3. Unzip it.
4. Double-click **Launch.bat**.
5. First run installs numpy + FLAC support. Then **Browse** and open your recording.

Or from a terminal in this folder:

```
python InfraExplorer.py
```

## DSP self-test

```
python InfraExplorer.py --selftest
```

Preserve first. Measure second. Translate third. Enhance only for listening.
