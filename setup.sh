cd asr
pip install -r requirements.txt
git clone --recursive https://github.com/parlance/ctcdecode.git
cd ctcdecode && pip install .

gdown https://drive.google.com/uc?id=1fA3MNHDkO-ThK2tEIng_o6VXstXtF93I

mkdir final_model
mv model_best-6.pth final_model
cp hw_asr/configs/ds2-librispeech-train.json final_model
