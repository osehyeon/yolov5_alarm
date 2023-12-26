import os
import urllib.request
from pycocotools.coco import COCO

# 데이터 및 애노테이션 파일의 경로 설정
data_dir = "./knife_image"  # 이 디렉토리에 COCO 데이터셋을 저장합니다.
ann_dir = os.path.join(data_dir, 'annotations')
ann_file_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
ann_file = os.path.join(ann_dir, 'instances_val2017.json')

# 애노테이션 파일 다운로드
if not os.path.exists(ann_file):
    if not os.path.exists(ann_dir):
        os.makedirs(ann_dir)

    ann_zip = os.path.join(ann_dir, 'annotations_trainval2017.zip')
    urllib.request.urlretrieve(ann_file_url, ann_zip)
    print("Annotations downloaded.")
    
    import zipfile
    with zipfile.ZipFile(ann_zip, 'r') as zip_ref:
        zip_ref.extractall(ann_dir)
    os.remove(ann_zip)
    print("Annotations extracted.")

# COCO 객체 초기화
coco = COCO(ann_file)

# 'knife' 카테고리의 이미지 ID 가져오기
cat_ids = coco.getCatIds(catNms=['knife'])
img_ids = coco.getImgIds(catIds=cat_ids)

# 이미지 정보 로드
imgs = coco.loadImgs(img_ids)
images_dir = os.path.join(data_dir, 'val2017')

# 'knife' 카테고리의 모든 이미지 다운로드
for img_info in imgs:
    img_url = img_info['coco_url']
    img_filename = os.path.join(images_dir, img_info['file_name'])
    
    if not os.path.exists(img_filename):  # 이미지가 아직 다운로드되지 않았다면
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)
        urllib.request.urlretrieve(img_url, img_filename)
        print(f"Downloaded {img_info['file_name']}")

print("Download completed!")
