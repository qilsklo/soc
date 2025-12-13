# soc
we take riedon output and make a csv that works for our scripts. then we use soc-ocv curves from the battery datasheet + kalman filter to correct coulomb counts. 
The current datasheet is based on INR-18650-M35A cells.


how to try it out:
- get some test data from your pack, it should look something like the file `rsfday1raw.txt`
- set the filenames on the last line of `parse_riedon_savefile.py` to the correct ones
- then run `kalman3.py` with the correct -i and -o flags
- Yay you have soc now! try comparing the first and last rows of the ekf% against the cc% to view the correction
- use `analyze_soc_results.py` if you want a graph comparing the EKF and raw CC SoCs over time
