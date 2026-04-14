# Machine Vision Intermediate Project
Computer vision pipeline for automated morphological classification of eucalyptus and pine seedlings, developed as part of the Visão de Máquinas course at Insper.

The system extracts key quality attributes from seedling images captured against a blue background, replacing manual inspection with an objective, scalable, and traceable routine.

Measured attributes:

- Total plant height (bounding-box approach and stem-path tracing)
- Collar diameter
- Total leaf area (color segmentation)
- Leaf count (connected-component analysis)

Tech stack: Python · OpenCV · NumPy
Dataset: Eucalyptus and pine seedlings photographed against a controlled blue backdrop under uniform lighting conditions.
