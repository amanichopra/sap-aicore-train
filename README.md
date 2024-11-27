## Using AI Core to Train a Model for Pose Estimation

This track introduces you to using AI Core to deploy your own custom training workflows. We will train a simple neural network to classify yoga poses given an input image. We will use a dataset called [Yoga-82](https://sites.google.com/view/yoga-82/home) that has images and labels for 82 different yoga poses. The images are first converted to pose embeddings using a pose detection model called [MoveNet](https://www.tensorflow.org/hub/tutorials/movenet). The model takes an input image and outputs a 17-dimensional vector representing the locations of 17 keypoints of the body. After generating the MoveNet pose embeddings, we will pass these into a simple 4-layer feed-forward network (defined in `model_utils.py`) to classify the pose embedding into 1 of the 82 yoga pose labels. The preprocessing of generating MoveNet embeddings is already done for you, and the data is located in `data/pose_embeddings.csv`. There is also metadata containing the labels and statistics about the image's pixel values (located in `data/metadata.csv`). An example of inference on a video is located in `model/inference.mp4`.

### Prerequisites

1. Follow [this](https://developers.sap.com/tutorials/hcp-create-trial-account..html
) trial to make a BTP trial account and follow [this](https://developers.sap.com/tutorials/appstudio-onboarding..html) tutorial to use Business Application Studio.
2. Open up a terminal session at the root of the cloned repository. It is highly suggested to create a virtual Python environment for following along to avoid conflicts with other packages that could be already installed on your system. A virtual environment can be created using `python3 -m venv .venv`. Activate the environment in your current terminal session using `source .venv/bin/activate`. Make sure that all subsequent steps are executed within the context of this newly created virtual environment.
3. Install the requirements using `pip install -r requirements.txt`. We will be using the Python [SDK](https://pypi.org/project/ai-core-sdk/) for AI Core. 
4. Download the `aicore_creds.json`, `docker_creds.json`, `gh_creds.json`, and `s3_creds.json` that are shared with you and store them in this project directory.

### Deploying a Training Workflow

Open the notebook `deploy_training_workflow.ipynb` and follow the steps.
