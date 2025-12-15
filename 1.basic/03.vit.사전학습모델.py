# huggingface transformer vit
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import BytesIO
import os, json
from glob import glob

"""
ViT 사전학습 모델 비교: Hugging Face vs timm

=== 전체 스토리 ===
같은 ViT 모델이지만 두 가지 다른 라이브러리로 사용할 수 있습니다.

1. Hugging Face Transformers
   - 자연어 처리(NLP)로 유명한 라이브러리
   - 사용하기 쉽고, 전처리가 자동화됨
   - 모델과 프로세서가 세트로 제공

2. timm (PyTorch Image Models)
   - 이미지 모델 전문 라이브러리
   - 다양한 모델 변형 제공
   - 더 많은 커스터마이징 가능

오늘은 같은 이미지로 두 라이브러리를 비교해봅니다!
"""

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


def use_huggingface_vit():
    '''
    Hugging Face Transformers로 ViT 모델 로드
    
    === 스토리 ===
    Hugging Face는 마치 "올인원 패키지"같습니다.
    모델 다운로드 → 전처리 설정 → 클래스 라벨까지
    모든 것이 자동으로 준비됩니다!
    
    마치 조리 완료된 밀키트를 받는 것처럼,
    바로 사용할 수 있게 모든 게 준비되어 있습니다.
    '''
    try:
        from transformers import ViTForImageClassification, ViTImageProcessor
        
        print("=" * 60)
        print("[Hugging Face ViT 모델 로드]")
        print("=" * 60)
        
        # ============================================================
        # 모델과 프로세서 로드
        # ============================================================
        # model_name: Hugging Face Hub에 있는 모델의 주소
        # "google/vit-base-patch16-224"의 의미:
        #   - google: Google이 학습시킨 모델
        #   - vit-base: ViT Base 크기 (86M 파라미터)
        #   - patch16: 16×16 패치 사용
        #   - 224: 224×224 이미지 입력
        #
        # 이 코드를 처음 실행하면:
        # 1. 인터넷에서 모델 다운로드 (약 350MB)
        # 2. 로컬 캐시에 저장 (다음부터는 빠름)
        # 3. 메모리에 로드
        model_name = 'google/vit-base-patch16-224'
        print(f"\n모델 다운로드 중: {model_name}")
        print("(처음 실행 시 시간이 걸립니다)")
        
        # ViTImageProcessor: 이미지 전처리 담당
        # - 자동으로 224×224 리사이즈
        # - ImageNet 통계로 정규화
        # - 텐서 변환
        # 우리가 할 일: 이미지만 넣으면 됨!
        processor = ViTImageProcessor.from_pretrained(model_name)
        
        # ViTForImageClassification: 분류 모델
        # - 이미지 입력 → 1000개 클래스 확률 출력
        # - ImageNet 클래스 라벨 내장
        model = ViTForImageClassification.from_pretrained(model_name)
        
        # eval() 모드:
        # - 학습 모드 OFF
        # - Dropout 비활성화
        # - BatchNorm 고정
        # 추론할 때는 항상 eval() 필요!
        model.eval()
        
        # ============================================================
        # 모델 상세 정보 출력
        # ============================================================
        print(f'\n[모델 정보]')
        
        # 파라미터 수: 모델의 "학습 가능한 숫자들"의 개수
        # ViT-Base는 약 86M (8천6백만) 개
        total_params = sum(p.numel() for p in model.parameters())
        print(f'  파라미터 수: {total_params:,} ({total_params/1e6:.1f}M)')
        
        # config: 모델의 설정 정보 (설계도)
        print(f'  입력 채널 수: {model.config.num_channels}')  # RGB = 3
        print(f'  이미지 크기: {model.config.image_size}×{model.config.image_size}')  # 224×224
        print(f'  패치 크기: {model.config.patch_size}×{model.config.patch_size}')  # 16×16
        print(f'  히든 크기: {model.config.hidden_size}')  # 768 (임베딩 차원)
        print(f'  레이어 수: {model.config.num_hidden_layers}')  # 12개 Transformer 블록
        print(f'  어텐션 헤드 수: {model.config.num_attention_heads}')  # 12개 헤드
        print(f'  클래스 수: {len(model.config.id2label)}')  # 1000개 (ImageNet)
        
        # id2label: 숫자 → 클래스 이름 매핑
        # 예: 281 → "tabby cat"
        #     207 → "golden retriever"
        
        return model, processor
        
    except Exception as e:
        print(f'✗ Hugging Face ViT 로드 실패: {e}')
        return None, None


def use_timm_vit():
    '''
    timm 라이브러리로 ViT 모델 로드
    
    === 스토리 ===
    timm은 "이미지 모델 백화점"입니다.
    수백 가지 사전학습 모델을 제공하고,
    더 세밀한 설정이 가능합니다.
    
    Hugging Face가 "쉬운 자동화"라면,
    timm은 "전문가용 도구"에 가깝습니다.
    '''
    try:
        import timm
        
        print("\n" + "=" * 60)
        print("[timm ViT 모델 로드]")
        print("=" * 60)
        
        # ============================================================
        # 사용 가능한 모델 목록 확인
        # ============================================================
        # timm.list_models(): timm에서 제공하는 모든 모델 검색
        # 'vit*': 이름에 'vit'가 들어간 모델만
        # pretrained=True: 사전학습된 가중치가 있는 것만
        #
        # ViT 변형들:
        # - vit_tiny: 작은 모델 (5M)
        # - vit_small: 중간 모델 (22M)
        # - vit_base: 기본 모델 (86M) ← 우리가 사용
        # - vit_large: 큰 모델 (307M)
        # - vit_huge: 매우 큰 모델 (632M)
        vit_models = timm.list_models('vit*', pretrained=True)
        print(f'\n사용 가능한 ViT 모델 (일부):')
        for model_name in vit_models[:5]:  # 처음 5개만 출력
            print(f'  - {model_name}')
        print(f'  ... (총 {len(vit_models)}개 모델)')
        
        # ============================================================
        # 모델 생성
        # ============================================================
        # timm.create_model():
        # - 모델 아키텍처 생성
        # - pretrained=True: 학습된 가중치 다운로드
        # 
        # 'vit_base_patch16_224':
        # - base: 86M 파라미터
        # - patch16: 16×16 패치
        # - 224: 224×224 입력
        print(f'\n모델 로드 중: vit_base_patch16_224')
        model = timm.create_model('vit_base_patch16_224', pretrained=True)
        model.eval()
        
        print(f"✓ 모델 로드 완료")
        
        # ============================================================
        # 데이터 전처리 설정
        # ============================================================
        # timm은 전처리를 직접 설정해야 함
        # (Hugging Face는 자동이었음)
        #
        # data_config: 모델에 맞는 전처리 설정
        # - input_size: 입력 크기
        # - mean: 정규화 평균
        # - std: 정규화 표준편차
        # - interpolation: 리사이즈 방법
        data_config = timm.data.resolve_model_data_config(model)
        
        # create_transform: 실제 전처리 파이프라인 생성
        # is_training=False: 평가 모드 (증강 없음)
        transform = timm.data.create_transform(**data_config, is_training=False)
        
        return model, transform
        
    except Exception as e:
        print(f'✗ timm ViT 로드 실패: {e}')
        print('💡 설치 방법: pip install timm')
        return None, None


def classify_image_hf(model, processor, image):
    """
    Hugging Face 모델로 이미지 분류
    
    === 스토리 ===
    1. 이미지를 processor에 넣으면 자동 전처리
    2. 전처리된 데이터를 모델에 입력
    3. 1000개 클래스에 대한 점수(logits) 출력
    4. Softmax로 확률 변환
    5. 가장 높은 확률 Top-5 출력
    
    마치 사진을 찍어서 "이게 뭐야?"라고 물어보면
    "85% 확률로 고양이, 10% 확률로 호랑이..." 답하는 것!
    """
    
    if model is None:
        print("  ✗ 모델이 로드되지 않았습니다.")
        return None
    
    # ============================================================
    # [1단계] 이미지 전처리
    # ============================================================
    # processor(images=image, return_tensors="pt"):
    # - images=image: PIL Image 객체 입력
    # - return_tensors="pt": PyTorch 텐서로 반환
    #
    # 내부에서 일어나는 일:
    # 1. 이미지 리사이즈 (224×224)
    # 2. 정규화: (pixel - mean) / std
    # 3. [0, 255] → [-2, 2] 범위로 변환
    # 4. (H, W, C) → (C, H, W) 차원 변경
    # 5. 배치 차원 추가 [1, 3, 224, 224]
    inputs = processor(images=image, return_tensors="pt")
    
    print(f"\n[전처리된 입력]")
    print(f"  pixel_values shape: {inputs['pixel_values'].shape}")  # [1, 3, 224, 224]
    # 1: 배치 크기 (이미지 1장)
    # 3: RGB 채널
    # 224×224: 이미지 크기
    
    # ============================================================
    # [2단계] 모델 추론
    # ============================================================
    # torch.no_grad(): gradient 계산 안 함 (메모리 절약)
    # 추론할 때는 역전파가 필요 없으므로!
    with torch.no_grad():
        # model(**inputs): 딕셔너리 언패킹
        # inputs = {'pixel_values': tensor}
        # → model(pixel_values=tensor)
        outputs = model(**inputs)
    
    # logits: 원시 출력 점수 (확률 아님!)
    # shape: [1, 1000]
    # - 1: 배치
    # - 1000: ImageNet 클래스 개수
    #
    # logits의 의미:
    # - 높은 값: 모델이 확신
    # - 낮은 값: 모델이 의심
    # 예: [2.5, -1.3, 0.8, ...]
    logits = outputs.logits
    
    print(f"\n[모델 출력]")
    print(f"  logits shape: {logits.shape}")  # [1, 1000]
    
    # ============================================================
    # [3단계] 확률로 변환
    # ============================================================
    # F.softmax(): logits → 확률
    # dim=-1: 마지막 차원(1000개 클래스)에 대해
    #
    # Softmax 공식:
    # p_i = exp(logit_i) / sum(exp(logit_j))
    #
    # 결과:
    # - 모든 확률의 합 = 1.0
    # - 각 값은 0~1 사이
    # 예: [0.85, 0.05, 0.03, ...]
    probs = F.softmax(logits, dim=-1)
    
    # ============================================================
    # [4단계] Top-5 추출
    # ============================================================
    # torch.topk(probs, 5):
    # - 가장 높은 확률 5개를 찾기
    # - 반환: (확률값, 인덱스)
    #
    # 예:
    # top5_probs = [0.85, 0.08, 0.03, 0.02, 0.01]
    # top5_indices = [281, 282, 283, 207, 285]
    top5_probs, top5_indices = torch.topk(probs, 5)
    
    # ============================================================
    # [5단계] 결과 출력
    # ============================================================
    print(f"\n[Top-5 예측 결과]")
    for i, (prob, idx) in enumerate(zip(top5_probs[0], top5_indices[0])):
        # id2label: 인덱스 → 클래스 이름 변환
        # 281 → "tabby cat"
        # 207 → "golden retriever"
        label = model.config.id2label[idx.item()]
        
        # 출력 형식:
        # 1. tabby cat              : 85.23%
        # 2. tiger cat              :  8.14%
        print(f"  {i+1}. {label:30s}: {prob.item()*100:6.2f}%")
    
    return top5_probs[0], top5_indices[0]


def classify_image_timm(model, transform, image):
    """
    timm 모델로 이미지 분류
    
    === 스토리 ===
    Hugging Face와 거의 비슷하지만,
    전처리를 직접 transform으로 해야 합니다.
    
    또한 클래스 이름을 인터넷에서 가져와야 함
    (timm은 모델만 제공, 라벨은 별도)
    """
    
    if model is None:
        print("  ✗ 모델이 로드되지 않았습니다.")
        return None
    
    # ============================================================
    # [1단계] 이미지 전처리
    # ============================================================
    # transform(image): PIL Image → 텐서
    # 내부 처리:
    # 1. Resize & CenterCrop
    # 2. ToTensor: [0, 255] → [0, 1]
    # 3. Normalize: (x - mean) / std
    #
    # 결과: [3, 224, 224]
    img_tensor = transform(image)
    
    # unsqueeze(0): 배치 차원 추가
    # [3, 224, 224] → [1, 3, 224, 224]
    img_tensor = img_tensor.unsqueeze(0)
    
    print(f"\n[전처리된 입력]")
    print(f"  tensor shape: {img_tensor.shape}")
    
    # ============================================================
    # [2단계] 모델 추론
    # ============================================================
    with torch.no_grad():
        # timm 모델은 직접 텐서를 받음
        # (Hugging Face는 딕셔너리였음)
        outputs = model(img_tensor)
    
    # outputs: [1, 1000] logits
    # Hugging Face와 동일한 형식
    print(f"\n[모델 출력]")
    print(f"  outputs shape: {outputs.shape}")
    
    # ============================================================
    # [3단계] 확률 변환 & Top-5
    # ============================================================
    probs = F.softmax(outputs, dim=-1)
    top5_probs, top5_indices = torch.topk(probs, 5)
    
    # ============================================================
    # [4단계] 클래스 라벨 가져오기
    # ============================================================
    # timm은 클래스 이름을 제공하지 않음!
    # ImageNet 라벨을 인터넷에서 다운로드해야 함
    try:
        # GitHub에 있는 ImageNet 클래스 파일
        url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        response = requests.get(url, timeout=10)
        
        # 1000줄의 텍스트 파일
        # 각 줄 = 클래스 이름
        # 예:
        # 0: tench
        # 1: goldfish
        # ...
        # 281: tabby cat
        categories = [s.strip() for s in response.text.splitlines()]
        
        print(f"\n[Top-5 예측 결과]")
        for i, (prob, idx) in enumerate(zip(top5_probs[0], top5_indices[0])):
            # categories[인덱스] = 클래스 이름
            label = categories[idx.item()] if idx.item() < len(categories) else f"class_{idx.item()}"
            print(f"  {i+1}. {label:30s}: {prob.item()*100:6.2f}%")
            
    except Exception as e:
        # 인터넷 연결 실패 시 인덱스만 출력
        print(f"\n[Top-5 예측 결과 (인덱스)]")
        for i, (prob, idx) in enumerate(zip(top5_probs[0], top5_indices[0])):
            print(f"  {i+1}. class_{idx.item():4d}: {prob.item()*100:6.2f}%")
    
    return top5_probs[0], top5_indices[0]


if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    print("\n" + "=" * 70)
    print(" " * 20 + "ViT 모델 비교 테스트")
    print(" " * 15 + "Hugging Face vs timm")
    print("=" * 70)
    
    # ============================================================
    # [1단계] 모델 로드
    # ============================================================
    # 두 가지 다른 라이브러리로 같은 ViT-Base 모델 로드
    # - Hugging Face: 자동화, 쉬움
    # - timm: 전문가용, 유연함
    print("\n[1단계] 모델 로드")
    hf_model, hf_processor = use_huggingface_vit()
    timm_model, timm_transform = use_timm_vit()
    
    # ============================================================
    # [2단계] 테스트 이미지 준비
    # ============================================================
    # glob: 디렉토리에서 패턴에 맞는 파일 찾기
    # '*.jpg': jpg 확장자를 가진 모든 파일
    file_paths = r'C:\Users\sally\OneDrive\문서\GitHub\MultiModal\1.basic\download_img'
    files = glob(os.path.join(file_paths, '*.jpg'))
    
    if not files:
        print(f"\n✗ 이미지 파일을 찾을 수 없습니다: {file_paths}")
    else:
        print(f"\n✓ 총 {len(files)}개 이미지 발견")
        for f in files:
            print(f"  - {os.path.basename(f)}")
        
        # ============================================================
        # [3단계] Hugging Face 모델로 추론
        # ============================================================
        print("\n" + "=" * 70)
        print("[2단계] Hugging Face 모델로 추론")
        print("=" * 70)
        
        for idx, file in enumerate(files):
            print("\n" + "-" * 60)
            print(f"📷 이미지 {idx+1}/{len(files)}: {os.path.basename(file)}")
            print("-" * 60)
            
            # PIL Image로 로드
            # .convert('RGB'): 모든 이미지를 RGB로 통일
            # (RGBA, 흑백 등 다양한 형식 대응)
            test_img = Image.open(file).convert('RGB')
            print(f"이미지 크기: {test_img.size}")
            
            # Hugging Face 모델로 예측
            if hf_model is not None:
                classify_image_hf(hf_model, hf_processor, test_img)
        
        # ============================================================
        # [4단계] timm 모델로 추론
        # ============================================================
        if timm_model is not None:
            print("\n" + "=" * 70)
            print("[3단계] timm 모델로 추론")
            print("=" * 70)
            
            for idx, file in enumerate(files):
                print("\n" + "-" * 60)
                print(f"📷 이미지 {idx+1}/{len(files)}: {os.path.basename(file)}")
                print("-" * 60)
                
                test_img = Image.open(file).convert('RGB')
                print(f"이미지 크기: {test_img.size}")
                
                # timm 모델로 예측
                classify_image_timm(timm_model, timm_transform, test_img)
    
    print("\n" + "=" * 70)
    print("✓ 모든 테스트 완료!")
    print("=" * 70)
    print("\n💡 결과 비교:")
    print("  - Hugging Face: 자동 전처리, 클래스 라벨 내장")
    print("  - timm: 수동 전처리, 다양한 모델 선택")
    print("  - 두 모델의 예측 결과는 거의 동일해야 합니다!")