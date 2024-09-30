# Seismic Activity Analyzer

## NASA Space Apps 2024 Challenge: Seismic Detection Across the Solar System
Planetary seismology missions struggle with the power requirements necessary to send continuous seismic data back to Earth. But only a fraction of this data is scientifically useful! Instead of sending back all the data collected, what if we could program a lander to distinguish signals from noise, and send back only the data we care about? Your challenge is to write a computer program to analyze real data from the Apollo missions and the Mars InSight Lander to identify seismic quakes within the noise!

### Our Approach
Pre-train a machine learning model using the provided training datasets, and test the model with the challenge dataset (to be revealed during the hacking period of the hackathon). This also includes data cleaning, and preprocessing using a Jupyter notebook and various different Python libraries (details to follow).

### Challenge Background
Research in planetary seismology is fundamentally constrained by a lack of data due to the difficulty of transferring high-resolution seismic signals back to Earth. The amount of power required to transmit data scales with distance, so the further a target body is from Earth, the more energy is required to transmit the same amount of data. Quakes are typically rare events, meaning that although large amounts of continuous data are recorded and sent back to Earth, only a small fraction of this data contains useful signals. This constraint is especially important as seismologists will likely be sharing the lander with science teams from other disciplines who have different objectives and instruments, some of which may be transferring even larger amounts of data to Earth. Consequently, data is recorded at lower resolution or with fewer instruments than might be optimal to achieve the desired science.

A potential solution for this issue is to run algorithms on a lander to differentiate seismic data from the noise, so that only the useful signals can be extracted and sent back to Earth. This is tricky to do in practice, as seismic signals on other planets tend to look different than on Earth, and the signal might be only faintly observable in the noise.


### Project Contributors
Starbound Voyagers Team: Adam Lam, Asmi Gujral, Robel Gebreselasse, Mya Mckinnon
<!--![alt text](Assets/TeamPicture.jpeg)--!>