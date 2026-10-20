# Emotion Detection using Transfer Learning

This is a small project where I trained a model to recognise 7 facial emotions (angry, disgust, fear, happy, neutral, sad, surprise) and then ran it live on my webcam.

I used MobileNetV2 (pretrained on ImageNet) instead of training a CNN from scratch, and trained it on the FER-2013 dataset from Kaggle.

## How it works

1. Load the images from the `train` and `test` folders (resized to 96x96).
2. Take MobileNetV2 and freeze it, then add my own small classifier on top (pooling, dropout, a 7-class softmax layer). Data augmentation (flip, rotation, zoom) is applied during training.
3. **Phase A:** train only the new classifier layer for 15 epochs.
4. **Phase B:** unfreeze the last 30 layers of MobileNetV2 and fine-tune for 15 more epochs with a very small learning rate.
5. For the webcam demo, OpenCV's Haar cascade finds the face, the face is cropped and resized, and the model predicts the emotion. It also shows a bar for every emotion.

## Dataset

FER-2013: 28,709 training images and 7,178 test images. The dataset is not included in this repo, you can download it from Kaggle.

The classes are very imbalanced:

| Emotion | Training images |
|---|---|
| happy | 7,215 |
| neutral | 4,965 |
| sad | 4,830 |
| fear | 4,097 |
| angry | 3,995 |
| surprise | 3,171 |
| disgust | 436 |

"Happy" has around 16 times more images than "disgust", so the model sees much less of the rare classes. I did not use class weights in this version.

## Results

| | Accuracy |
|---|---|
| After Phase A (frozen base) | about 43% |
| After Phase B (fine-tuned) | **51.9%** |

Fine-tuning improved accuracy by around 9 points. 51.9% is not very high, but it is well above random guessing (about 14% for 7 classes). FER-2013 is also a hard dataset: the images are small (48x48), many labels are noisy, and even humans don't always agree on the emotion.

Things to keep in mind:

- I used the `test` folder as the validation set while training, so there is no separate untouched test set. The final number may be a bit optimistic.
- Because of the class imbalance, the overall accuracy hides how the model does on each emotion. I haven't checked per-class results or a confusion matrix yet, but I expect "disgust" to be the weakest.
- The accuracy was still going up at epoch 15, so training longer would probably help.
- In the webcam demo it works best with good lighting and a face looking straight at the camera. The predictions can flicker from frame to frame.
- It only reads facial expressions, it can't tell what someone actually feels.


## Running it

Install the libraries with `pip install -r requirements.txt`. Download FER-2013 and put it 
in a folder called `data/` with `train` and `test` subfolders (one folder per emotion), 
then run the notebook cell by cell. The last cell opens the webcam (`q` to quit).

## Files

- `Emotion_Detection_transfer_learning.ipynb`: the full notebook (training, results and webcam demo)
- `emotion_model.keras`: the trained model
- `class_names.json`: the order of the emotion labels
- `requirements.txt`

## Things I want to try next

- Class weights or oversampling to deal with the imbalance
- A proper train/validation/test split
- A confusion matrix and per-class scores
- Training for more epochs and trying other models like EfficientNet
- Smoothing the predictions in the webcam demo so they don't flicker

## Credits

- FER-2013 dataset on Kaggle
- MobileNetV2 from `keras.applications`
- OpenCV Haar cascade for face detection
