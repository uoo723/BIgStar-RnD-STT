**목차**

- [대스타 R\&D](#대스타-rd)
  - [개요](#개요)
  - [데이터 전처리](#데이터-전처리)
    - [Mel Frequency Cepstral Coefficient (MFCC)](#mel-frequency-cepstral-coefficient-mfcc)
      - [Fast Fourier Trasnform (FFT)](#fast-fourier-trasnform-fft)
      - [Short Term Fourier Transform (STFT)](#short-term-fourier-transform-stft)
      - [Mel Filter Bank](#mel-filter-bank)
  - [모델](#모델)
    - [Wav2Vec 2.0](#wav2vec-20)
  - [데이터셋](#데이터셋)
    - [KsponSpeech](#ksponspeech)
    - [Zeroth-Korean](#zeroth-korean)
  - [평기지표](#평기지표)
  - [Instruction](#instruction)
    - [디렉토리 구조](#디렉토리-구조)
    - [데이터셋 준비](#데이터셋-준비)
      - [KsponSpeech](#ksponspeech-1)
      - [Zeroth-Korean](#zeroth-korean-1)
    - [모델 훈련](#모델-훈련)
      - [TL;DR](#tldr)
      - [KsponsSpeech 훈련 (Pre-training)](#ksponsspeech-훈련-pre-training)
      - [Zeorth-Korean 훈련 (Fine-tuning)](#zeorth-korean-훈련-fine-tuning)
      - [Command](#command)

# 대스타 R&D

## 개요

- **음성 텍스트 변환 (Speech to Text)** 은 인간의 음성 인터페이스를 텍스트 데이터로 추출해내는 기술.
- 음성 데이터로 인식 네트워크 모델을 생성하여 학습하는 단계와 음성을 인식하는 탐색 단계로 나뉨.
- 디코더(Decoder)는 기존의 음성 데이터 지식을 사용하여 문자정보로 해석하여 출력하 는 역할을 함. 디코딩 단계에서는 음향 모델, 언어 모델, 발음 사전을 이용하여 입력된 벡터를 모델과 비교하여 최종 결정함.
- 최근에는 seq2seq 기반으로 하는 모델을 많이 사용함.

## 데이터 전처리

디지털 음성 신호를 머신러닝의 feature로 사용하기 위해서는 적절한 전처리 과정이 필요하다. 이 과정에서 음성 신호에 있는 노이즈와 불필요한 정보를 제거하고 중요한 feature만 남게 된다.

### Mel Frequency Cepstral Coefficient (MFCC)

음성 인식 분야에서 쓰이는 여러 전처리 기법 중 가장 널리 쓰이는 기법은 MFCC이다. 

**변환 과정**
  1. 입력 음성을 20ms~40ms (음소 단위) 사이의 짧은 구간으로 나눈다. 그리고 이러한 구간을 프레임 (Frame) 이라고 한다.
  2. 이렇게 나누어진 프레임을 푸리에 변환 (Fourier Transform, FT)을 수행하여 프레임에 담겨 있는 주파수 정보를 추출한다. 이 결과를 스펙트럼 (spectrum)이라 한다.
  3. 스펙트럼에 인간의 청각기관 성질을 이용한 Mel Filter Bank를 적용하여 Mel Spectrum를 뽑는다.
  4. Mel Spectrum에 log를 취한 후 역 푸리에 변환 (Inverse Fourier Transform, IFT)를 수행하여 MFCC를 얻는다.

| ![framework](https://user-images.githubusercontent.com/7765506/223026271-cfcf4455-4dbb-45ef-9dcc-12aea8af6805.png) |
| :--: |
| *MFCC 변환 과정* [[출처](https://ratsgo.github.io/speechbook/docs/fe/mfcc)] |

#### Fast Fourier Trasnform (FFT)

- 음성 신호는 각 주파수별 신호의 합으로 구성.
- 원 신호에 어떤 정보가 있는지 파악하기 위해 주파수별 신호의 세기를 분해하는 작업 필요.
- 이를 수행하는 알고리즘이 푸리에 변환 (Fourier Transform).
- FFT는 주기성과 대칭성을 이용하여 이산 푸리에 변환 (Discrete Fourier Transform, DFT)과 그 역변환을 빠르게 수행하는 알고리즘.
- 푸리에 변환을 통해 음성 신호를 frequency, magnitude, phase 도메인으로 분해하게 됨.

| ![original_voice](https://user-images.githubusercontent.com/7765506/223003423-ac17f6c8-37db-4c5f-9fb5-88585cab6ce9.jpg) |
| :--: |
| *원본 음성* |

| ![FFT_transformed_voice](https://user-images.githubusercontent.com/7765506/223005502-21328b05-fe00-4846-99a2-d38ee81984e0.png) |
| :--: |
| *FFT 변환 결과* |

#### Short Term Fourier Transform (STFT)

- 음성 신호는 시계열 정보이므로 시간 정보를 반영해주는 것이 적합.
- 주로 특정 시간 구간마다 window를 씌워 FFT을 수행하는 것이 일반적.
- 이를 Short Term Fourier Trasnform (STFT)라 함.
- 1차원 벡터인 푸리에 변환값이 2차원 matrix가 됨.

| ![spectrogram](https://user-images.githubusercontent.com/7765506/223007234-90782a8d-460d-4112-b0d2-abd1b74fdf96.jpg) |
| :--: |
| *STFT로 변환 후 출력한 Spectrogram (log scale)* |

#### Mel Filter Bank

- STFT 변환으로 생성된 matrix의 frequency 차원이 매우 크기 때문에 frequency 구간 별로 정보를 압축하여 feature를 추출해야 함.
- Feature를 추출할 때 인간의 청각기관의 성질을 모방.
- 인간의 청각기관은 음성을 인식할 때 낮은 주파수에 더 민감하게 반응.
- 이러한 성질에 기반한 filter를 Mel Filter Bank라 함. (Mel Scale)
- Mel Filter Bank는 낮은 주파수 대역에서 더 세밀하게 정보를 압축.
- Mel Filter Bank를 통과한 feature는 Mel Spectrum이라 함.

| ![mel_filter_bank](https://user-images.githubusercontent.com/7765506/223008062-71ec6573-e5fc-4f23-9d58-008b80b66863.jpg) |
| :--: |
| *Mel Filter Bank* |

| ![mfcc_visualization_result](https://user-images.githubusercontent.com/7765506/223291525-41e1afe4-b65a-4084-9a97-b35f33ff561d.jpg) |
| :--: |
| *MFCC 변환 결과* |

## 모델

### Wav2Vec 2.0

**Wav2Vec**은 raw waveform을 입력으로 하기 때문에 위의 데이터 전처리 과정을 생략할 수 있음.

- 53,000 시간의 label이 없는 데이터로 representation training을 한 후, 10분의 label된 데이터로 훈련하여 음성 인식기를 만들 수 있는 모델.
- labeling되어 있지 않은 다량의 데이터로 representation을 학습한 후, 소량의 labeling 데이터로 fine-tuning 함.
- 음성 오디오만으로 강력한 표현을 만들어 낼 수 있어 labeling이 적은 데이터셋에 대해서도 충분히 좋은 성능의 모델을 만들 수 있음.
- Constrastive Learning: Latent speech representation과 context representation을 비슷하게 되도록 학습하게 됨.
- Bi-directional contextualized representation: 현재 위치를 masking하고 주변의 데이터로부터 masking된 위치를 유추할 수 있는 Transformer 구조.
- Vector quantized targets: Gumbel softmax 방식으로 latent speech representation에서 가장 영향을 많이 미치는 vector 추출.
- Self-training: 1) Raw wavform을 다중 CNN encoder에 넣어 25ms 길이의 latent vector로 변환. 2) Encoder를 통과한 latent vector는 양자화 모듈 (quantizer)에서 이산화됨. 3) Transformer에서 전체 오디오 sequence의 정보가 추가됨. masking된 입력을 Transformer에 넣으면 주변 정보를 이용하여 복원된 context representation이 생성됨. Transformer의 출력은 constrastive task를 수행하는 데에 사용됨.

| ![wav2vec](https://user-images.githubusercontent.com/7765506/223291935-04f72271-bd75-4d4f-8f04-e5fa5d3cc3c8.jpg) |
| :--: |
| Wav2Vec 2.0 모델 구조 | 

## 데이터셋

<table>
  <tr>
    <td>Datasets</td>
    <td># of training</td>
    <td># of test</td>
  </tr>
  <tr>
    <td>KsponSpeech</td>
    <td>622,545</td>
    <td>-</td>
  </tr>
  <tr>
    <td>Zeroth-Korean</td>
    <td>22,263</td>
    <td>457</td>
  </tr>
</table>

### KsponSpeech

한국지능정보사회진흥원의 개발 데이터 구축 사업인 AI Hub로부터 제공 받음.

[데이터셋다운로드](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=123)

### Zeroth-Korean

[Kaldi](http://kaldi-asr.org/)를 이용하여 구축하는 한국어 음성인식 오픈소스 프로젝트.

[데이터셋 다운로드](http://www.openslr.org/40)

## 평기지표

$$CER = \frac{S_c + D_c + I_c}{N_c} = \frac{S_c + D_c + I_c}{S_c + D_c  + C_c}$$

$$WER = \frac{S_w + D_w + I_w}{N_w} = \frac{S_w + D_w + I_w}{S_w + D_w  + C_w}$$

- $CER$: Character Error Rate
- $WER$: Word Error Rate
- $S_c$ ($S_w$): 잘못 대체된 음절 (단어) 수
- $D_c$ ($D_w$): 잘못 삭제된 음절 (단어) 수
- $I_c$ ($I_w$): 잘못 추가된 음절 (단어) 수
- $C_c$ ($C_w$): 맞춘 음절 (단어) 수
- $N_c$ ($N_w$): 참조 음절 (단어) 수

## Instruction

### 디렉토리 구조

```bash
├── main.py
├── preprocess.py
├── train.py
├── scripts
│   ├── run_preprocess.sh
│   ├── run_train.sh
│   ├── run_wav2vec_kspon.sh
│   ├── run_wav2vec_zeroth.sh
│   ├── run_test.sh
├── src
│   ├── base_trainer.py
│   ├── callbacks.py
│   ├── optimizers.py
│   ├── utils.py
│   ├── dataset
│   │   ├── __init__.py
│   │   ├── kspon
│   │   │   ├── __init__.py
│   │   │   └── preprocess
│   │   │       ├── __init__.py
│   │   │       ├── base.py
│   │   │       └── character.py
│   │   └── zeroth_korean
│   │       ├── __init__.py
│   │       └── zeroth_korean.py
│   ├── wav2vec
│   │   ├── __init__.py
│   │   └── trainer.py
├── data
│   ├── kspon_speech
│   │   ├── original
│   │   │   └── ...
│   │   ├── preprocessed
│   │   │   └── transcripts.csv
│   │   │
│   ├── huggingface
│   │   └── ...
├── logs
│   ├── 1
│   │   ├── "run_id"
│   │   │   ├── checkpoints
│   │   │   │   ├── "epoch=xx-cer=xx.ckpt"
└── └── └── └── └── "last.ckpt"
```

### 데이터셋 준비

#### KsponSpeech

1. [AI Hub](https://aihub.or.kr/) 로그인 후 한국어 데이터 중 "한국어 음성" 데이터 다운로드.

<img width="1453" alt="스크린샷 2023-03-13 10 38 27" src="https://user-images.githubusercontent.com/7765506/224589114-1891573b-2f17-4e8b-9808-f57888a07844.png">

2. Root 프로젝트 하위 디렉토리 `data/kspon_speech/original`에 다운로드 받은 데이터셋 압축해제하여 저장.

```bash
├── ...
├── data
│   ├── kspon_speech
│   │   ├── original
│   │   │   ├── KsponSpeech_01
│   │   │   │   ├── KsponSpeech_0001
│   │   │   │   │   ├── KsponSpeech_000001.pcm
│   │   │   │   │   ├── KsponSpeech_000001.txt
│   │   │   │   │   ├── KsponSpeech_000002.pcm
│   │   │   │   │   ├── KsponSpeech_000002.txt
│   │   │   │   │   └── ...
│   │   │   │   ├── KsponSpeech_0002
│   │   │   │   └── ...
│   │   │   ├── KsponSpeech_02
│   │   │   └── ...
│   │   └── ...
│   └── ...
└── ...
```

3. 전처리 스크립트 실행

```bash
./scripts/run_preprocess.sh
```

#### Zeroth-Korean

[HuggingFace Dataset](https://huggingface.co/docs/datasets/index) API를 이용하여 자동으로 데이터셋을 다운로드하고 전처리함.

### 모델 훈련

#### TL;DR

Pre-training (KsponSpeech) + Fine-tuning (Zeorth-Korean) 

```bash
./scripts/run_train.sh
```

#### KsponsSpeech 훈련 (Pre-training)

```bash
./scripts/run_wav2vec_kspon.sh [seed]
```

#### Zeorth-Korean 훈련 (Fine-tuning)

```bash
./scripts/run_wav2vec_zeorth.sh [seed]
```

#### Command

```bash
$ python main.py train-wav2vec --help
Usage: main.py train-wav2vec [OPTIONS]

  Train Wav2Vec

Options:
  Train Options:
    --mode [train|test|predict]   train: train and test are executed. test:
                                  test only, predict: make prediction
                                  [default: train]
    --skip-test                   If set to true, skip test after training
                                  [default: False]
    --run-id TEXT                 MLFlow Run ID for resume training
    --save-run-id-path PATH       Path for saving run id path
    --model-name TEXT             Model name  [required]
    --dataset-name TEXT           Dataset name  [default: kspon;required]
    --valid-size FLOAT|INT        Validation dataset size  [default: 0.2]
    --seed INTEGER                Seed for reproducibility  [default: 0]
    --swa-warmup INTEGER          Warmup for SWA. Disable: 0  [default: 10]
    --mp-enabled                  Enable Mixed Precision  [default: False]
    --early INTEGER               Early stopping step  [default: 10]
    --reset-early                 Reset early  [default: False]
    --load-only-weights           Load only weights not all training states
                                  [default: False]
    --load-best                   Load best model instead of last model when
                                  training is resumed  [default: False]
    --load-last                   Load last model instead of best model in
                                  test mode  [default: False]
    --early-criterion [cer|wer|loss]
                                  Early stopping criterion  [default: cer]
    --eval-step INTEGER           Evaluation step during training  [default:
                                  100]
    --num-epochs INTEGER          Total number of epochs  [default: 40]
    --train-batch-size INTEGER    Batch size for training  [default: 8]
    --test-batch-size INTEGER     Batch size for test  [default: 1]
    --no-cuda                     Disable cuda  [default: False]
    --num-workers INTEGER         Number of workers for data loader  [default:
                                  4]
    --lr FLOAT                    learning rate  [default: 0.001]
    --decay FLOAT                 Weight decay  [default: 0.01]
    --accumulation-step INTEGER   accumlation step for small batch size
                                  [default: 1]
    --gradient-max-norm FLOAT     max norm for gradient clipping
    --model-cnf PATH              Model config file path
    --data-cnf PATH               Data config file path
    --optim-name [adamw|sgd]      Choose optimizer  [default: adamw]
    --scheduler-warmup FLOAT|INT  Ratio of warmup among total training steps
    --use-deepspeed               Use deepspeed to reduce gpu memory usage
                                  [default: False]
    --scheduler-type [linear|cosine|cosine_with_restarts|polynomial|constant|constant_with_warmup]
                                  Set type of scheduler
  Log Options:
    --log-dir PATH                log directory  [default: ./logs]
    --tmp-dir PATH                Temp file directory  [default: ./tmp]
    --log-run-id                  Log run id to tmp dir  [default: False]
    --experiment-name TEXT        experiment name  [default: baseline]
    --run-name TEXT               Set Run Name for MLFLow
    --tags <TEXT TEXT>...         Set mlflow run tags  [default: ]
    --run-script PATH             Run script file path to log
  Dataset Options:
    --dataset-filepath PATH       Preprocessed data filepath  [default: ./data
                                  /kspon_speech/preprocessed/transcripts.csv]
    --cache-dir PATH              Cache directory for huggingface datasets
                                  [default: ./data/huggingface/datasets]
  Wav2Vec Options:
    --pretrained-model-name TEXT  Pretrained model name  [default:
                                  kresnik/wav2vec2-large-xlsr-korean]
    --use-pretrained-model        Use pretrained model  [default: False]
    --num-hidden-layers INTEGER   # of hidden layers in the Transformer
                                  [default: 12]
    --num-attention-heads INTEGER
                                  # of attention heads in the Transformer
                                  [default: 12]
    --intermediate-size INTEGER   Dimensionality of the intermediate layer in
                                  the Transformer  [default: 3072]
    --hidden-size INTEGER         Dimensionality of the encoder layers and the
                                  pooler layer in the Transformer  [default:
                                  768]
    --ctc-loss-reduction [mean|sum]
                                  Specifies the reduction to apply to the
                                  output of `torch.nn.CTCLoss`  [default: sum]
    --ctc-zero-infinity           Whether to zero infinite losses and the
                                  associated gradients of `torch.nn.CTCLoss`
                                  [default: False]
  --help                          Show this message and exit.  [default:
                                  False]
```
