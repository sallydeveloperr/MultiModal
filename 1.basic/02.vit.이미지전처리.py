import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from typing import List

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 전역 변수
IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def load_existing_images(save_dir: str) -> List:
    '''기존 이미지 로드'''
    print(f'\n[이미지 로드]')
    
    if not os.path.exists(save_dir):
        print(f'  ✗ 디렉토리가 없습니다: {save_dir}')
        return []
    
    downloaded_images = []
    
    for filename in ['cat.jpg', 'dog.jpg', 'bird.jpg']:
        filepath = os.path.join(save_dir, filename)
        if os.path.exists(filepath):
            print(f'  ✓ 발견: {filename}')
            downloaded_images.append(filepath)
        else:
            print(f'  ✗ 없음: {filename}')
    
    print(f'\n  총 {len(downloaded_images)}개 이미지 로드 완료')
    return downloaded_images


def basic_image_loading(image_path: str):
    '''기본 이미지 로딩 방법'''
    img = Image.open(image_path)
    
    print(f'\n[이미지 정보]')
    print(f'  파일명: {os.path.basename(image_path)}')
    print(f'  이미지 모드: {img.mode}')
    print(f'  이미지 크기: {img.size}')
    
    # numpy 배열로 변환
    img_array = np.array(img)
    print(f'\n[Numpy 배열 변환]')
    print(f'  배열 shape: {img_array.shape}')  # (H, W, C)
    print(f'  데이터 타입: {img_array.dtype}')
    print(f'  값 범위: {img_array.min()} ~ {img_array.max()}')
    
    return img


def vit_standard_preprocessing(img):
    '''ViT 표준 전처리 파이프라인'''
    
    # 전처리 파이프라인
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])
    
    print(f'\n[전처리 결과]')
    print(f'  원본 이미지 크기: {img.size}')
    
    img_tensor = preprocess(img)
    print(f'  전처리 후 shape: {img_tensor.shape}')
    print(f'  전처리 후 값 범위: {img_tensor.min():.3f} ~ {img_tensor.max():.3f}')
    
    # 배치 차원 추가
    img_batch = img_tensor.unsqueeze(0)
    print(f'  배치 처리 후 shape: {img_batch.shape}')
    
    return img_tensor, preprocess


def training_augmentation(img):
    """학습 시 사용하는 데이터 증강"""   
    
    # 학습용 증강 파이프라인
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    
    # 평가용 파이프라인 (증강 없음)
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    
    print(f"\n[학습용 증강 기법]")
    print(f"  1. RandomResizedCrop: 랜덤 위치/크기로 자르기")
    print(f"  2. RandomHorizontalFlip: 50% 확률로 좌우 반전")
    print(f"  3. ColorJitter: 밝기, 대비, 채도 변형")
    print(f"  4. RandomRotation: ±15도 회전")
    
    # 같은 이미지에 여러 번 증강 적용
    print(f"\n[동일 이미지에 증강 적용 예시]")
    augmented_images = []
    for i in range(4):
        aug_img = train_transform(img)
        augmented_images.append(aug_img)
        print(f"  증강 {i+1}: shape={aug_img.shape}, "
              f"min={aug_img.min():.3f}, max={aug_img.max():.3f}")
    
    return train_transform, val_transform, augmented_images


def visualize_preprocess(img, img_tensor, augmented_images, filename='preprocess'):
    '''전처리 과정 시각화'''
    
    def denormalize(tensor, mean=MEAN, std=STD):
        '''정규화 역변환'''
        tensor = tensor.clone()
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
        return torch.clamp(tensor, 0, 1)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 원본 이미지
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('원본 이미지', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # 전처리된 이미지
    img_display = denormalize(img_tensor)
    axes[0, 1].imshow(img_display.permute(1, 2, 0).numpy())
    axes[0, 1].set_title('전처리 완료 (224×224)', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # 패치 분할 시각화
    patch_size = 16
    img_np = img_display.permute(1, 2, 0).numpy()
    axes[0, 2].imshow(img_np)
    
    # 그리드 그리기
    for i in range(0, 224, patch_size):
        axes[0, 2].axhline(y=i, color='red', linewidth=1, alpha=0.7)
        axes[0, 2].axvline(x=i, color='red', linewidth=1, alpha=0.7)
    
    axes[0, 2].set_title('패치 분할 (16×16)\n총 196개 패치', fontsize=14, fontweight='bold')
    axes[0, 2].axis('off')
    
    # 증강된 이미지들
    for i, aug_img in enumerate(augmented_images[:3]):
        aug_display = denormalize(aug_img)
        axes[1, i].imshow(aug_display.permute(1, 2, 0).numpy())
        axes[1, i].set_title(f'데이터 증강 {i+1}', fontsize=14, fontweight='bold')
        axes[1, i].axis('off')
    
    plt.suptitle(f'ViT 이미지 전처리 및 증강 - {filename}', fontsize=18, fontweight='bold')
    plt.tight_layout()
    
    # 저장
    save_path = f'02.vit.preprocess_{filename}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'\n✓ 시각화 저장: {save_path}')
    
    plt.show()


if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    print("=" * 60)
    print("ViT 이미지 전처리 실습")
    print("=" * 60)
    
    # 이미지 로드 - 절대 경로로 변경
    image_dir = r'C:\Users\sally\OneDrive\문서\GitHub\MultiModal\1.basic\download_img'
    sample_images = load_existing_images(image_dir)
    
    if not sample_images:
        print("\n⚠ 이미지가 없습니다. 프로그램을 종료합니다.")
    else:
        # 각 이미지 처리
        for idx, img_path in enumerate(sample_images):
            print("\n" + "=" * 60)
            print(f"[이미지 {idx+1}/{len(sample_images)}] 처리 중")
            print("=" * 60)
            
            # 이미지 로딩
            img = basic_image_loading(img_path)
            
            # 전처리
            img_tensor, preprocess = vit_standard_preprocessing(img)
            
            # 증강
            train_transform, val_transform, augmented_images = training_augmentation(img)
            
            # 시각화
            print("\n[시각화 생성 중...]")
            filename = os.path.splitext(os.path.basename(img_path))[0]
            visualize_preprocess(img, img_tensor, augmented_images, filename)
        
        print("\n" + "=" * 60)
        print("✓ 모든 이미지 전처리 완료!")
        print("=" * 60)