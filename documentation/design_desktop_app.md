# DRAW Single Machine Workflow

DRAW can be used a Desktop App as well!

## Architecture

![draw-desktop-app-architecture](https://ik.imagekit.io/oj8f972s8/a9t/a9t-desktop-app-arch.drawio.png)

1. **DICOM Files Placement:**
   - Place DICOM files in the designated input folder. This can be automatically done from Scanning Service or manually placed.
2. **Automatic Processing:**
   - The system automatically detects new files in the input folder.
   - Sends the detected DICOM files to the NNUNet model.
3. **NNUNet Prediction:**
   - The NNUNet model analyzes the received images.
   - Generates predictions for each relevant area within the images.
4. **Combine Overlapping Predictions:**
   - Predictions from areas covered by multiple images are combined or merged.
   - Creates a single, unified prediction for those areas.
5. **Prediction Export:**
   - Combined predictions are exported to the specified output folder in the designated format.

## How to run

Change the value of `DICOM_WATCH_DIR` in `draw/config.py` and run by any of the following in the command line:

```bash
python main.py start-pipeline #Preferred way
```

or

```bash
python run.py #Will be deprecated in the future
```

## Key Components

- **NNUNet Model:** Trained model for prediction
- **DICOM Files:** Standard format for storing medical images, such as MRI scans or CT scans. Files come from scanning service
- **Input Folder:** Contains DICOM files to be processed
- **Output Folder:** Stores exported predictions.
- **SQL Database:** Tracks the status of predictions. Acts as a Queue.

## Additional Notes

- Enables multiple predictions to be executed simultaneously by multiprocessing. **2** models will be run in parallel
- Batch size of processing is **1**
- Specifications of Prediction Server:
  - Intel® i7
  - Nvidia® Quadro T1000 8GB
  - 32 GB RAM

## FAQ

### Why use a database as a queue?

In this low-traffic scenario, a database is a convenient choice for both queuing and analytics.

- **Streamlines management:** It keeps everything in one place, making it easier to track prediction status and results.
- **Direct access for analytics:** You can easily access prediction data for analysis, helping you understand trends and improve the process.
- **Efficiency for low traffic:** It's well-suited for environments with lower prediction volumes.
- **Doubles as storage:** It serves as both a queue for managing tasks and a storage system for prediction data.

This two-in-one approach simplifies the overall architecture and promotes efficient data analysis.
