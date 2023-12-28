# Using A9T as a Desktop App

## Architecture

![a9t-desktop-app-architecture](https://ik.imagekit.io/oj8f972s8/a9t/a9t-desktop-app-arch.drawio.png)

## How to run

A9T can be used a Desktop App as well! Change the value of `DICOM_WATCH_DIR` in `a9t/config.py` and run by any of the following in the command line:

```bash
python main.py start-pipeline #Preferred way
```

or

```bash
python run.py #Will be deprecated in the future
```

## Facts

- Batch size of processing is `1`
- `2` models will be run in parallel
