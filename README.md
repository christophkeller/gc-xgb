# XGBoost emulator for GEOS-Chem
Example code for developing a machine learning emulator for the GEOS-Chem chemistry model, following the methodology described in Keller and Evans (2019). This python script trains a chemistry emulator for a specified species based on a preprocessed set of training data. The test data is available at https://gmao.gsfc.nasa.gov/gmaoftp/geoscf/gc-xgb/svm and will be downloaded automatically if missing. Tested with Python 3.6.7 and 3.6.9.

Usage: `python gcxgb.py`

**References:**

Keller, C. A. and Evans, M. J.: Application of random forest regression to the calculation of gas-phase chemistry within the GEOS-Chem chemistry model v10, Geosci. Model Dev., 12, 1209–1225, https://doi.org/10.5194/gmd-12-1209-2019, 2019.
