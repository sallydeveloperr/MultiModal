import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import sys, os

#1. 패치분할
def patch_embedding():
    '''
    이미지를 패치로 분할하는 과정
    
    === 전체 스토리 ===
    우리는 지금 224×224 크기의 컬러 사진을 작은 퍼즐 조각들로 나누려고 합니다.
    마치 큰 그림을 16×16 크기의 작은 조각들로 잘라서, 각 조각을 숫자로 설명하는 것과 같습니다.
    최종적으로 각 조각은 768개의 숫자로 표현됩니다.
    '''
    # 설정
    image_size = 224        # 원본 사진의 크기 (가로 224픽셀 × 세로 224픽셀)
    patch_size = 16         # 퍼즐 한 조각의 크기 (16×16 픽셀)
    channels = 3            # RGB 컬러 사진이므로 3개 층 (빨강, 초록, 파랑)
    embedding_dim = 768     # 각 퍼즐 조각을 768개의 숫자로 요약할 예정
    
    # [2단계] 퍼즐 조각이 몇 개 나올지 계산하기
    # ============================================================
    # 224픽셀을 16픽셀씩 자르면 가로로 14조각 (224 ÷ 16 = 14)
    # 세로도 마찬가지로 14조각
    # 따라서 총 14 × 14 = 196개의 퍼즐 조각이 나옵니다
    # 
    # 시각적으로 보면:
    # ┌────────────────┐
    # │ □□□□□□□□□□□□□□ │ ← 가로 14개
    # │ □□□□□□□□□□□□□□ │
    # │ □□□□□□□□□□□□□□ │  
    # │ ...            │ ← 세로 14개
    # └────────────────┘
    # 총 196개의 □ (퍼즐 조각)


    # 패치수 계산
    num_patches = (image_size // patch_size) ** 2
    print(f'이미지 크기 : {image_size} x {image_size}')
    print(f'패치 크기 : {patch_size} x {patch_size}')
    print(f'채널 수 : {channels}')
    print(f'패널 수 : {image_size // patch_size} x {image_size // patch_size}')

    # 더미 이미지 생성 (계산을 위해 샘플 이미지 생성)
    # [3단계] 실제로 작업할 사진 준비하기 (더미 이미지 생성)
    # ============================================================
    # torch.randn: 임의의 숫자로 채워진 가짜 이미지를 만듭니다
    # shape [1, 3, 224, 224]의 의미:
    #   - 1: 사진 1장 (배치 크기)
    #   - 3: RGB 3개 색상 층
    #   - 224: 세로 224픽셀
    #   - 224: 가로 224픽셀
    dummy_image = torch.randn(1, channels, image_size, image_size)  #배치 1개
    print(f'더미 이미지 생성')
    print(f'입력 이미지 shape : {dummy_image.shape}')   # [1, 3, 224, 224]
    # 여기까지가 준비과정

    # patch 사이즈만큼 분할 -> cnn의 컨볼루션을 사용하여
    # 패치분할(Conv2d를 사용)
    # Conv2d : stride(1이면 옆으로 도장을 찍는 것) = patch_size를 stride와 동일하게 해주면 겹치지 않는 patch 추출 가능
    # 방법 1: Conv2d로 패치 임베딩 (효율적)
    # [4단계] 마법의 가위 준비하기 - Conv2d
    # ============================================================
    # Conv2d는 이미지를 자동으로 잘라주는 "스마트 가위"입니다
    
    # 파라미터 설명:
    # - in_channels=3: RGB 3개 층을 모두 봅니다
    # - out_channels=768: 각 조각을 768개 숫자로 변환합니다
    # - kernel_size=16: 16×16 크기로 자릅니다
    # - stride=16: 16칸씩 점프합니다 (겹치지 않게)
    #
    # stride가 중요한 이유:
    # - stride= 1이면: 1칸씩 이동 → 조각들이 겹쳐버림 ❌
    # - stride=16이면: 16칸씩 이동 → 딱 붙어있지만 안 겹침 ✓
    #
    # 동작 과정:
    # Step 1: 왼쪽 위 (0,0)에서 16×16 조각 추출
    #   ┌──────┐
    #   │16×16 │ ← 첫 번째 조각
    #   └──────┘
    # Step 2: 오른쪽으로 16칸 점프 (0,16)
    #           ┌──────┐
    #           │16×16 │ ← 두 번째 조각
    #           └──────┘
    # Step 3: 계속 반복하여 196개 조각 모두 추출
    patch_embed = nn.Conv2d(in_channels=channels, out_channels=embedding_dim, kernel_size=patch_size, stride = patch_size)

    # 패치 임베딩을 적용
    # [5단계] 실제로 이미지 자르기 - Conv2d 적용
    # ============================================================
    # patch_embed(더미 이미지)를 실행하면:
    # 
    # 입력: [1, 3, 224, 224]
    #       └─ 1장의 RGB 사진, 224×224 크기
    # 
    # ⬇️ Conv2d 마법 발동 ⬇️
    # 
    # 출력: [1, 768, 14, 14]
    #       └─ 1장의 사진이
    #          768개 채널(각 조각의 768차원 표현)
    #          14×14 위치(196개 조각의 위치)로 변환됨
    #
    # 무슨 일이 일어났나요?
    # - 원래 224×224 사진을 16×16씩 잘라서 14×14=196조각으로 만들었습니다
    # - 각 조각(원래 16×16×3=768개 픽셀)을 768개 숫자로 압축했습니다
    # 
    # [1, 768, 14, 14]에서 각 위치의 의미:
    # - 위치 [0, :, 0, 0]: 왼쪽 위 조각의 768개 숫자
    # - 위치 [0, :, 0, 1]: 그 옆 조각의 768개 숫자
    # - 위치 [0, :, 13, 13]: 오른쪽 아래 조각의 768개 숫자
    patches = patch_embed(dummy_image)
    print(f'\n패치 임베딩 후')
    print(f'Conv2d 후 shape : {patches.shape}')     # (1, 768, 14, 14)

    # Flatten : (batch, channels, height, width) → (batch, num_patches, embedding_dim)
    # [6단계] 격자 형태를 일렬로 펼치기 - Flatten & Transpose
    # ============================================================
    # 지금은 [1, 768, 14, 14] 형태 = 14×14 격자에 768차원 벡터들이 배치
    # 
    # 우리가 원하는 최종 형태: [1, 196, 768]
    # = 196개 조각이 일렬로, 각각 768차원
    #
    # Step 1: flatten(2) - 뒤의 두 차원(14, 14)을 하나로 합치기
    #   [1, 768, 14, 14] → [1, 768, 196]
    #   14×14 격자를 → 196개 일렬로 펼침
    #
    # Step 2: transpose(1,2) - 1번과 2번 차원의 순서 바꾸기
    #   [1, 768, 196] → [1, 196, 768]
    #   (배치, 특징차원, 조각들) → (배치, 조각들, 특징차원)
    #
    # 왜 이렇게 바꾸나요?
    # Transformer는 이런 형식을 좋아합니다:
    # [배치, 시퀀스 길이, 특징 차원]
    # 
    # 문장 처리와 비교하면:
    # 문장: "나는 학교에 간다" (4개 단어)
    #       [배치, 4개 단어, 각 768차원 임베딩]
    # 
    # 우리 이미지: 196개 퍼즐 조각
    #       [배치, 196개 조각, 각 768차원 임베딩]
    #
    # 결과적으로:
    # [
    #   [0.23, -1.45, 0.87, ..., 2.1],   ← 조각 1 (768개 숫자)
    #   [1.34, 0.56, -0.23, ..., 1.5],   ← 조각 2 (768개 숫자)
    #   ...
    #   [-0.67, 1.23, 0.45, ..., 0.8]    ← 조각 196 (768개 숫자)
    # ]

    # 이런 형태가 됩니다!
    patches_flat = patches.flatten(2).transpose(1,2)
    print(f'flatten 후 shape : {patches_flat.shape}')   #[1, 196, 768]

    # 각 패치는 768차원의 벡터가 된다.
    print(f'\n패치 수 : {patches_flat.shape[1]}')  
    print(f'각 패치의 임베딩 차원 수 : {patches_flat.shape[2]}')  
    return patches_flat

    # 정리하면:
    # 1. 224×224 컬러 사진 1장으로 시작
    # 2. 16×16 크기로 잘라서 196조각 생성
    # 3. 각 조각을 768개 숫자로 표현
    # 4. 최종적으로 196개의 768차원 벡터 완성!
    #
    # 이제 이 196개 벡터를 Transformer에 넣으면
    # "이 사진이 강아지인지 고양이인지" 판단할 수 있습니다!

# 위치 임베딩의 역할
def positional_embedding():
    '''
    위치 임베딩 (Positional Embedding)
    
    === 전체 스토리 ===
    Transformer는 순서를 모릅니다!
    
    예를 들어 "나는 학교에 간다"라는 문장이 있을 때,
    Transformer는 [나는, 학교에, 간다]가 어떤 순서인지 모릅니다.
    "학교에 나는 간다"든 "간다 나는 학교에"든 구분 못합니다.
    
    마찬가지로 이미지의 196개 퍼즐 조각도:
    - 조각1이 왼쪽 위에 있는지
    - 조각50이 가운데에 있는지
    - 조각196이 오른쪽 아래에 있는지
    Transformer는 모릅니다!
    
    그래서 각 조각에 "위치 태그"를 달아줍니다.
    마치 좌석번호가 적힌 이름표를 달아주는 것처럼요.
    '''
    num_patches = 196
    embedding_dim = 768

    # 위치 임베딩 생성
    # 이 텐서는 학습대상 Optimizer에 의해 업데이트
    # 나중에 사용할 때:
    # 패치 임베딩 + 위치 임베딩 = 최종 임베딩
    # "이게 강아지 귀 조각이야" + "왼쪽 위에 있어" = "왼쪽 위의 강아지 귀"
    position_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embedding_dim)) # +1은 CLS 토큰
    print(f'위치 임베딩 shape : {positional_embedding.shape}')
    print(f'총 위치 수 : {num_patches+1} (패치 196 + cls 토큰 1)')
    # 배치차원 제거 : 각 위치를 하나의 벡터로 다루기 위해 배치크기가 1인 형태는 분석시 불필요
    # 왜 제거하나?
    # - 배치 차원은 "여러 이미지"를 동시 처리할 때 필요
    # - 지금은 위치 임베딩 자체를 분석하려는 것
    # - 197개 위치의 768차원 벡터만 있으면 충분
    pos_emb = positional_embedding.squeeze(0)
    # 코사인 유사도 계산을 위해서 정규화는 필수
    # 코사인 유사도 분석을 통해:
    # - 인접 위치가 비슷한 임베딩을 가지는지 확인 가능
    # - 모델이 2D 이미지 구조를 잘 학습했는지 평가 가능
    pos_emb_norms = pos_emb / pos_emb.norm(dim=1, keepdim=True)     # 단위벡터로 나눔
    # 코사인 유사도 행렬
    similarity = torch.mm(pos_emb_norms, pos_emb_norms.t())
    print(f'유사도 행렬 shape : {similarity.shape}')    # [197,197]
    # 학습전이라서 랜덤이지만 학습후에는 인접 패치끼리 유사해짐
    return position_embedding

# CLS 토큰
# CLS = Classification Token
    # - BERT에서 빌려온 개념
    # - 전체 이미지를 대표하는 특별한 토큰
    # - 최종 분류는 이 토큰의 출력값으로 수행
def cls_token():
    embedding_dim = 768
    num_patches = 196
    batch_size = 2      # 두개의 이미지
    # cls 토큰 생성
    cls_token = nn.Parameter(torch.rand(1,1,embedding_dim))     # 학습을 통해 이미지를 요약하는 벡터로 진화
    print(f'CLS 토큰 shape : {cls_token.shape}')    # [1,1,196]
    # 배치크기에 맞게 CLS 토큰 확장
    cls_tokens = cls_token.expand(batch_size, -1, -1)   # 배치마다 동일한 cls 토큰이 필요, expand는 view만 확장한 것
    print(f'배치확장 후 shape : {cls_tokens.shape}')    # [2,1,768]
    # 패치 임베딩 생성
    patch_embeddings = torch.randn(batch_size, num_patches, embedding_dim)
    # 패치된 임베딩 출력
    print(f'패치 임베딩 shape : {patch_embeddings.shape}')
    # cls 토큰을 시퀀스 맨 앞에 추가
    embeddings = torch.cat([cls_tokens, patch_embeddings], dim=1)
    # ┌─────┬────────────────────────────┐
    # │ CLS │ 조각1 조각2 ... 조각196     │
    # └─────┴────────────────────────────┘
    print(f'결합 후 shape : {embeddings.shape}')    #[2,197,768]
    return embeddings

# 4. self-attention
def self_attention():
    '''self-attention 매커니즘
    Multi-Head Self Attention의 구성요소
        Q(Query) K(Key) V(Value)
    '''
    embedding_dim = 768
    num_heads = 12
    head_dim = embedding_dim // num_heads
    seq_len = 197   #196패치 + 1 cls
    batch_size = 1
    # 더미 데이터 생성
    x = torch.rand(batch_size, seq_len, embedding_dim)
    # QKV 선형 레이어
    qkv_proj = nn.Linear(embedding_dim, embedding_dim*3)    # 하나의 Linear 연산으로 QKV 동시에 생성
    # QKV 계산(x)로 더미데이터를 넣어준다
    qkv = qkv_proj(x)
    print(f'QKV shape : {qkv.shape}')
    # Q K V 분리
    qkv = qkv.shape(batch_size, seq_len, 3, num_heads, head_dim) #[B, N, 3, heads, head_dim]
    qkv = qkv.permut(2,0,3,1,4)     #[3, B, heads, N, head_dim]
    q,k,v = qkv[0], qkv[1], qkv[2]
    print(f'Q shape : {q.shape}')   # 동일한 모양
    print(f'K shape : {k.shape}')   # 동일한 모양
    print(f'V shape : {v.shape}')   # 동일한 모양
    
    # Attention Score 계산
    scale = head_dim ** -0.5
    attn = (q @ k.transpose(-2, -1)) * scale
    print(f"\n[Attention Score]")
    print(f"  Attention shape: {attn.shape}")  # [1, 12, 197, 197]
    
    # Softmax
    attn = attn.softmax(dim=-1)
    print(f"  Softmax 후 합계 (각 행): {attn[0, 0, 0].sum().item():.4f}")  # 1.0
    
    # Value와 곱하기
    out = attn @ v
    print(f"\n[Attention 적용]")
    print(f"  출력 shape: {out.shape}")  # [1, 12, 197, 64]
    
    # 헤드 결합
    out = out.transpose(1, 2).reshape(batch_size, seq_len, embedding_dim)
    print(f"  헤드 결합 후: {out.shape}")  # [1, 197, 768]
    
    return attn



if __name__ == '__main__':
    patch_embedding()

