# DRAW: Autosegment

In the realm of Radiation Therapy Planning,
the significance of segmentation cannot be overstated.
Despite the existing availability of open-source models,
there is a notable absence of comprehensive solutions
that facilitate end-to-end segmentation for both organs at risk
and target volumes. To address this gap, we present the A9T framework,
leveraging the power of the nnUNet architecture.
This module is specifically tailored for seamless integration
into radiotherapy planning workflows.

## Features

- Seamlessly integrates with `DICOM` images, ensuring compatibility with standard medical imaging formats.
- Deals with Structure Overlap by splitting models
- Predicts both `Organs-At-Risk` and `Clinical Target Volumes`
- Works on multiple cancer sites:
  - `TSPrime`: Prostate Cancer Patients
  - `TSGyne`: Full Bladder Patients?
- Supports the execution of models in parallel using `multiprocessing`
- DRAW caters to the diverse needs of users by offering both automatic and manual segmentation options
- Database Integration for Monitoring and Analysing Workflows

## Code Documentation

For details about various commands, see [documentation](documentation)

## How to run

Incoming

## Acknowledgements and Future Plans

TODO
