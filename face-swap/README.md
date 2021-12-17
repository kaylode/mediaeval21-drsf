# Inference code for Face Swap

##### Reference paper for Face Swap: https://arxiv.org/pdf/2004.03234.pdf
##### This repo was modified from: https://github.com/AliaksandrSiarohin/motion-cosegmentation

# Setup: 
- Clone the repo: 
```
git clone https://github.com/??/face-swap
cd face-swap
```
- Install dependencies (you can use conda env if you want): 
```
pip install -r requirements.txt
```

- Download checkpoint (if the command below does not work, please download `vox-first-order.pth.tar` from [here](https://drive.google.com/drive/folders/1SsBifjoM_qO0iFzb8wLlsz_4qW2j8dZe) and put it into `./checkpoints` folder):
```
mkdir checkpoints
```
```
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1n2CqYEjM82X7sE40xrZpmnOxF6NekYW0' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1n2CqYEjM82X7sE40xrZpmnOxF6NekYW0" -O checkpoints/vox-first-order.pth.tar && rm -rf /tmp/cookies.txt
```
- Create save directory:
```
mkdir demo/save_swap
```

# Usage
- Run face swap
```
python face_swap.py --source_image "path_to_source_image" \
                    --target_video "path_to_target_video" 
```

- The face swap result:
  - The synthesis video will be saved at `demo/save_swap/result.mp4`
  - The comparison video will be saved at `demo/save_swap/output.mp4`
  - The synthesis video with audio will be saved at `demo/save_swap/ad-result.mp4`
  - The comparison video with audio will be saved at `demo/save_swap/ad-output.mp4`

  