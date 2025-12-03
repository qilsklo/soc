# soc
we take riedon output and make a csv that works for our scripts. then we use soc-ocv curves from the battery datasheet + kalman filter to correct coulomb counts. 


how to try it out:
- get some test data from your battery, it should look something like `save.txt`
- set the filename on the last line of `format_verify.py` by replacing `save.txt`
- then run evaluate_soc.py --> parsed_save_data_soc_results.csv comes out
- Yay you have soc now! try comparing the first and last rows of the ekf% against the cc% to view the correction
