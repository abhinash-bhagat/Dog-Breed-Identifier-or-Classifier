# Link to live Webapp: https://abhinash-bhagat-dog-breed-identifier--source-codedog-app-ic44kl.streamlit.app/

# 🐶 End-to-End Multiclass Dog Breed Classification

This notebook builds a multiclass image classifier using Tensorflow and Tensorflow Hub.

## 1. Problem
Identifying the breed of the dog by the given image.

## 2. Data
Provided with a training set and a test set of images of dogs. Each image has a filename that is its unique id. The dataset comprises 120 breeds of dogs. The list of breeds is as follows:
* affenpinscher
* afghan_hound
* african_hunting_dog
* airedale
* american_staffordshire_terrier
* appenzeller
* australian_terrier
* basenji
* basset
* beagle
* bedlington_terrier
* bernese_mountain_dog
* black-and-tan_coonhound
* blenheim_spaniel
* bloodhound
* bluetick
* border_collie
* border_terrier
* borzoi
* boston_bull
* bouvier_des_flandres
* boxer
* brabancon_griffon
* briard
* brittany_spaniel
* bull_mastiff
* cairn
* cardigan
* chesapeake_bay_retriever
* chihuahua
* chow
* clumber
* cocker_spaniel
* collie
* curly-coated_retriever
* dandie_dinmont
* dhole
* dingo
* doberman
* english_foxhound
* english_setter
* english_springer
* entlebucher
* eskimo_dog
* flat-coated_retriever
* french_bulldog
* german_shepherd
* german_short-haired_pointer
* giant_schnauzer
* golden_retriever
* gordon_setter
* great_dane
* great_pyrenees
* greater_swiss_mountain_dog
* groenendael
* ibizan_hound
* irish_setter
* irish_terrier
* irish_water_spaniel
* irish_wolfhound
* italian_greyhound
* japanese_spaniel
* keeshond
* kelpie
* kerry_blue_terrier
* komondor
* kuvasz
* labrador_retriever
* lakeland_terrier
* leonberg
* lhasa
* malamute
* malinois
* maltese_dog
* mexican_hairless
* miniature_pinscher
* miniature_poodle
* miniature_schnauzer
* newfoundland
* norfolk_terrier
* norwegian_elkhound
* norwich_terrier
* old_english_sheepdog
* otterhound
* papillon
* pekinese
* pembroke
* pomeranian
* pug
* redbone
* rhodesian_ridgeback
* rottweiler
* saint_bernard
* saluki
* samoyed
* schipperke
* scotch_terrier
* scottish_deerhound
* sealyham_terrier
* shetland_sheepdog
* shih-tzu
* siberian_husky
* silky_terrier
* soft-coated_wheaten_terrier
* staffordshire_bullterrier
* standard_poodle
* standard_schnauzer
* sussex_spaniel
* tibetan_mastiff
* tibetan_terrier
* toy_poodle
* toy_terrier
* vizsla
* walker_hound
* weimaraner
* welsh_springer_spaniel
* west_highland_white_terrier
* whippet
* wire-haired_fox_terrier
* yorkshire_terrier

The link to datasets: https://www.kaggle.com/c/dog-breed-identification/data
## 3. Evaluation
The evaluation is a file with prediction probability for each dog breed of each test image.
https://www.kaggle.com/competitions/dog-breed-identification/overview/evaluation

## 4. Features
Some information about the data:
* Dealing with images (unstructured data) so, We are using DL/TL.
* 120 breeds of dogs (120 different classes)
* 10000+ images in trainig sets (They have laebls.)
* 10000+ images in test sets (They don't have labels.)
