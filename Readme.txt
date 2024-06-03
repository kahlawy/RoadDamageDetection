Details instructions for Run Code! 

1. Download and install Spyder IDE from the official website: https://www.spyder-ide.org/

2. Download the dataset from Kaggle: https://www.kaggle.com/datasets/trolololo888/potholes-and-road-damage-with-annotations
   - You may need to create a Kaggle account if you don't have one already.
   - After downloading the dataset, extract the contents to a directory  on C:\kaggle\input\train and C:\kaggle\input\val divide 500 to 400 for train and 100 for validate.

3. Create a new Python script file (e.g., `create_env.py`) in Spyder IDE and add the following code to create a new conda environment and install the required libraries:
   ```python
   import os
   # Create a new conda environment (optional but recommended)
   os.system('conda create -n projEnv python=3.9 -y')
   os.system('conda activate projEnv')
   # Install the libraries using conda and pip
   os.system('conda install numpy opencv pytorch torchvision tqdm matplotlib -c pytorch -c conda-forge -y')
   os.system('pip install mean-average-precision albumentations')
   ```
   - Save the script file and run it in Spyder IDE to create the conda environment and install the libraries.

4. Open the `Road-damage-and-pothole-detection_KahlawyHussein_202301531.py` script in Spyder IDE.
   - Make sure to update the file paths in the script to match the location where you extracted the dataset.

5. Activate the conda environment in Spyder IDE:
   - Go to "Tools" -> "Preferences" in Spyder IDE.
   - In the Preferences window, navigate to "Python interpreter" under "Tools".
   - Click on the "Use the following Python interpreter" radio button.
   - Select the Python interpreter associated with your conda environment (`projEnv` in this example) from the dropdown list.
   - Click "Apply" and then "OK" to save the changes.

6. Run the `Road-damage-and-pothole-detection_KahlawyHussein_202301531.py` script in Spyder IDE.
   - The script will train the object detection model using the provided dataset.
   - During the training process, you can monitor the progress and see the training metrics in the Spyder IDE console.

7. After the training is complete, the script will save the trained model and generate output images in the specified output folder.
   - You can view the output images sample  for  detected road damages and potholes see directory images .

