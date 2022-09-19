
Steps to setup the pipeline:

1. Download the git repo into the personal system.

2. To generate the distorted images, scripts and the sample images are present in the folder “Data_gen”. please open the terminal under this folder and run the following commands -
	* For view distortion → python3 view.py <input_path> <output_path>
	* For ghosting + motion distortion → python3 ghosting_motion.py <input_path> <output_path>
	* For more ghosting distortion → python3 ghosting_more.py <input_path> <output_more_ghosting_path>
	* For motion distortion → python3 motion.py <input_path> <output_path>
	* For less ghosting distortion → python3 ghosting_less.py <input_path> <output_less_ghosting_path>
	* For missing patch distortion → python3 missing.py <input_path> <output_path>

3. To train and infer the model, upload the folders called “ACDC_Sample_Data”, “Unet_traditional”, “trained_model” and “Msc.ipynb” file to the gdrive. ACDC_Sample_Data folder has sample of ground truth and distorted images, Unet_traditional has traditional approach unet model and data, trained_model has final trained model (pix2pix, cycleGan, cycleMedGan and proposed) which can be used for inference and Msc.ipynb is a tool from which training and inference can be done.

4. Open the “Msc.ipynb” file in the google colab to train and infer the model

5. To train the model
	* Install dependencies as the cell indicates
	* Connect to gdrive
	* Specify the path to training source and target. Source will be ACDC_Sample_Data/Distorted and target will be ACDC_Sample_Data/Clean if training is correction model else ACDC_Sample_Data/Segmented_GT for proposed. The folder will be in gdrive. 
	* Specify the training mode as per which model needs to be trained :
		--> Train_mode → choose from [pix2pix,cyclegan,cyclemedgan,proposed]
		--> Model_mode → choose from [pix2pix,cycle_gan,cycle_mgan,proposed]
	* Model name and path is the user's choice. Specify the hyper-parameters
		--> Batch size = 2 or 4
		--> Learning rate = 0.0002
		--> Number of epochs 1500+ for better results
		--> Patch size 256
	* If continuing the training then specify the pretrained network path else ignore.
	* Just run the other cells to start the training process. 

6. To infer the model
	* Install dependencies as the cell indicates
	* Connect to gdrive
	* Directly jump to the “Generate prediction(s) from unseen dataset” cell.
	* Specify the source path “ACDC_Sample_Data/Distorted” which will be in gdrive. The result folder is the user's choice.
	* Specify the training mode as per which model needs to be trained :
		--> Train_mode → choose from [pix2pix,cyclegan,cyclemedgan,proposed]
		--> Model_mode → choose from [pix2pix,cycle_gan,cycle_mgan,proposed]
	* Specify the prediction folder path as per which model has to be inferred. For example to infer proposed → trained_model/pix2pix, which is in gdrive.
	* Specify the patch as 256, and checkpoint as latest for running inference.
	* To see the output image just execute the “Inspect the predicted output”.

7. To train and infer the traditional Unet model : open Unet.ipynb file in google colab. Mount to gdrive and change the directory to the Unet_traditional. Run the cells to start training. To infer the model, few modification needs to be done in main.py
	* In line 391 - change the saved_params_path to “models/unettrainval.pt”
	* Change load_model_params to true, save_model_params = false, train = false and test = true. 
	* Change the output saving directory to user’s choice gdrive path in line 287 under model_test function.

8. To validate the model output, some of the output images and ground truth are present in the ACDC_Sample_Data folder. The following commands can be executed in a personal system. 
	* To validate correction model → python3 SSIM.py <input_path_GT> <Input_Output>
	* To validate traditional model Unet output → python3 DiceScore.py <input_segmentedout> <input_maskGT>
	* To validate the proposed network → python3 proposedvalidation.py <segmentedoutput> <Segmented_GT>

9. To check the compactness of feature map through contrastive, please run the python code which is in ContrastiveVisual folder → python3 TSNE.py


Note: Please follow the steps to get smooth execution. If you need to reproduce the accuracy and results the model must be trained on a full ACDC dataset with specified parameters[important]. Although the detailed steps are given above, if any issues occur, please refresh and reconnect the colab to re-execution and try.
