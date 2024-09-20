## Demo
You can explore our demo by running `demo.ipynb`. This demonstration illustrates how our MA-LMM serves as a plug-and-play module that can be integrated into InstructBLIP seamlessly, requiring no fine-tuning for zero-shot evaluation.

## Requirements

You can install the conda environment by running:
```bash
pip install -e .
```

## Dataset
([Breakfast](https://serre-lab.clps.brown.edu/resource/breakfast-actions-dataset/), [COIN](https://coin-dataset.github.io/)).

[MSRVTT](https://github.com/xudejing/video-question-answering), [MSVD](https://github.com/xudejing/video-question-answering), and [ActivityNet](https://github.com/MILVLG/activitynet-qa).
For the video captioning task, we also conduct experiments on [Youcook2](http://youcook2.eecs.umich.edu/) dataset.


Then extract video frames of each video with fps=10.
   ```
    ├── data
        └── activitynet
            ├── annotation
            ├── frames
            ├── videos
        └── breakfast
            ├── annotation
            ├── frames
            ├── videos
        └── coin
            ├── annotation
            ├── frames
            ├── videos
        └── lvu
            ├── annotation
            ├── frames
            ├── videos
        └── msrvtt
            ├── annotation
            ├── frames
            ├── videos
        └── msvd
            ├── annotation
            ├── frames
            ├── videos
        └── youcook2
            ├── annotation
            ├── frames
            ├── videos
   ```
## Running

### Download Pre-trained LLM
We use Vicuna-v1.1 as our pre-trained LLM weights, you can download from this [link](https://github.com/lm-sys/FastChat/blob/main/docs/vicuna_weights_version.md) as arrange in this format.
   ```
   ├── llm
        ├── vicuna-7b
        ├── vicuna-13b
   ```
### Finetuning on Downstreaming Tasks
If you would like to fine-tune the model for various video datasets, please run the following command:
```bash
bash run_scripts/${dataset}/train.sh
```

### Testing
```bash
bash run_scripts/${dataset}/test.sh ${checkpoint_path}
```

## Acknowledgement
We referenced the repo below for the code
- [LAVIS](https://github.com/salesforce/LAVIS)



