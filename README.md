
# **Selected Topics in Visual Recognition using Deep Learning HW2**
StudentID: 311540015

Name: Do Huu Phu

## **Introduction**
This README provides details on the configurable parameters for HW2 training Digit Recognition task

Environment Setup:

- Python version: 3.8.17

- PyTorch version: 2.0.1
```bash
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2
```

Download the dataset and store it at the same level as the repository.





## **Data Parameters**
These parameters specify the directories where your training, validation, and test datasets are stored.

- `--config-file` (default: `data/train`):
  - **Description**: Path to the config data directory.

The settings for the experiment are available in the `config/HW2` folder and can be customized. For more details, refer to the [Detectron2 Config Documentation](https://detectron2.readthedocs.io/en/latest/modules/config.html).
  



## Example Usage

To train a model using the provided parameters, you can execute `run.bash` or the script as follows:

```bash
bash train.sh
```

To visualize: 
```bash
tensorboard --logdir log_folder